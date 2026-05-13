"""
Build a training/eval dataset from best-of-N simulator runs.

Walks a directory produced by ``python -m discoverllm.simulate.run --mode
best_of_n`` (each ``<artifact_id>/best_of_n.json``), extracts per-turn
candidate completions and rewards, optionally filters or re-weights them,
and saves the resulting rows as a HuggingFace dataset, JSON, or JSONL file.

Legacy ``synthesis.json`` filenames produced by older runs are still picked
up for back-compat.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from discoverllm.data._writers import save_conversations_jsonl, save_dataset
from discoverllm.pipeline.rewards import calculate_reward_and_deltas

# The "ours" system prompt that we bake into best-of-N conversations.
# Single source of truth lives in ``discoverllm.training.prompts``.
from discoverllm.training.prompts import SYSTEM_PROMPT_OURS as SYSTEM_PROMPT
from discoverllm.utils import count_tokens_in_conversation, is_distinct_response

REWARD_THRESHOLD = 0.05


def _recalculate_candidate_reward(
    candidate: Dict[str, Any],
    prev_criteria_objs: List[Dict[str, Any]],
    prompt: List[Dict[str, Any]],
) -> Dict[str, float]:
    """
    Recalculate reward for a candidate using stored criteria data.

    Args:
        candidate: The candidate dict with 'updated_criteria_objs', 'future_trajectory',
                   and 'assistant_response'
        prev_criteria_objs: Criteria state before this turn
        prompt: Conversation history up to (and including) the user message for this turn
                (used as fallback if future_trajectory is not available)

    Returns:
        Dict with 'reward', 'delta_awareness', 'delta_satisfaction'
    """
    new_criteria_objs = candidate.get("updated_criteria_objs")
    if not new_criteria_objs or not prev_criteria_objs:
        return {"reward": None, "delta_awareness": 0.0, "delta_satisfaction": 0.0}

    # Use chat_history from future_trajectory if available (preferred)
    future_trajectory = candidate.get("future_trajectory") or {}
    conversation_slice = future_trajectory.get("chat_history")

    # Fallback: build conversation slice from prompt + assistant_response
    if not conversation_slice:
        assistant_response = candidate.get("assistant_response", "")
        conversation_slice = prompt + [{"role": "assistant", "content": assistant_response}]

    result = calculate_reward_and_deltas(
        new_criteria_objs=new_criteria_objs,
        prev_criteria_objs=prev_criteria_objs,
        conversation_slice=conversation_slice,
    )
    return result


def _load_run_payload(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # pragma: no cover - defensive
        print(f"⚠️  Failed to load result file {path}: {e}")
        return None


def _get_prompt_for_turn(conversation_history: List[Dict[str, Any]], turn_index: int) -> List[Dict[str, Any]]:
    """
    In best_of_n runs the conversation is a flat list of messages:
    - Index 0: user (initial request) -> prompts turn 1
    - Index 1: assistant (turn 1 response)
    - Index 2: user (after turn 1) -> prompts turn 2
    - ...

    For turn N (1-based), the user message that prompts that turn is at index 2*(N-1).
    We include all messages up to and including that user message.
    """
    if not conversation_history:
        return []
    user_idx = 2 * (turn_index - 1)
    return conversation_history[: user_idx + 1] if user_idx < len(conversation_history) else conversation_history


def process_single_run(
    result_data: Dict[str, Any],
    artifact_id: str,
    conv_id: Optional[str] = None,
    max_token_length: int = 4096,
    minimum_gap: float = 0.0,
    score_type: str = "multiturn",
    add_labels: bool = False,
    is_filter_zeros: bool = False,
    recalculate_rewards: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
    """
    Process a single best_of_n.json payload and extract per-turn candidate rows.

    score_type:
        - 'multiturn': use candidate['full_results']['multiturn']['reward']
        - 'singleturn': use candidate['full_results']['singleturn']['reward']
        - 'direct': use candidate['reward'] (already-computed reward field)

    minimum_gap:
        - If > 0, require that the difference between the highest and lowest
          non-None candidate scores within a turn is at least this value.
          Otherwise the entire turn is skipped.

    recalculate_rewards:
        - If True, recalculate rewards from stored criteria data instead of using
          pre-computed rewards. Uses updated_criteria_objs from each candidate.

    Returns:
        rows: list of dicts with fields:
            - artifact_id
            - conv_id          (conversation ID, extracted from artifact_id for compatibility)
            - turn_id
            - prompt          (conversation history slice)
            - completion      (assistant response text)
            - score
            - criteria_history (full criteria history up to and including the state before this turn,
                                i.e., criteria_history[:turn_index] where turn_index is 1-based)
        results: per-turn mapping assistant_index -> score (used for summary stats)
        n_skipped_turns: number of turns skipped due to filtering
    """
    rows: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    n_skipped_turns = 0

    # Read the unified ConversationResult schema. The key is "conversation"
    # in the current schema; older synthesis.json files used the
    # "conversation_history" name — kept as a fallback for back-compat.
    conversation_history: List[Dict[str, Any]] = result_data.get(
        "conversation", result_data.get("conversation_history", [])
    )
    per_turn_candidates: List[Dict[str, Any]] = result_data.get("per_turn_candidates") or []
    criteria_history: List[List[Dict[str, Any]]] = result_data.get("criteria_history", [])

    for turn_entry in per_turn_candidates:
        turn_index = int(turn_entry.get("turn", 0))
        if turn_index <= 0:
            continue

        prompt = _get_prompt_for_turn(conversation_history, turn_index)
        candidates = turn_entry.get("candidates", [])
        if not candidates:
            n_skipped_turns += 1
            continue

        # Get criteria_history before the update for this turn
        # For turn N (1-based), the previous state is at index (N-1) in criteria_history
        # criteria_history[0] is the initial state, criteria_history[i] is state after turn i
        criteria_before_update = None
        if criteria_history and turn_index >= 1:
            prev_idx = turn_index - 1
            if prev_idx < len(criteria_history):
                # Store the full history up to and including the state before this turn
                # This provides context about how criteria evolved up to this point
                criteria_before_update = criteria_history[:prev_idx+1]

        curr_rows: List[Dict[str, Any]] = []
        curr_result: Dict[str, Any] = {}
        curr_delta_satisfactions: List[float] = []
        curr_delta_awareness: List[float] = []

        # Get previous criteria for recalculation (criteria state before this turn)
        # For turn N (1-based), the state before turn N is at index (N-1)
        prev_criteria_objs = None
        if recalculate_rewards and criteria_history and turn_index >= 1:
            prev_idx = turn_index - 1
            if prev_idx < len(criteria_history):
                prev_criteria_objs = criteria_history[prev_idx]

        for candidate in candidates:
            assistant_index = candidate.get("assistant_index")
            completion_text = candidate.get("assistant_response", "")
            if not completion_text:
                # If any candidate is empty, skip the whole turn — comparing
                # a partial set of completions would bias the preference pair.
                curr_rows = []
                break

            if add_labels:
                label_text = f"**ACTION = ASSISTANT_{assistant_index}**\n\n"
                completion_text = label_text + completion_text

            if recalculate_rewards and prev_criteria_objs is not None:
                # Recalculate reward from stored criteria data
                recalc_result = _recalculate_candidate_reward(
                    candidate=candidate,
                    prev_criteria_objs=prev_criteria_objs,
                    prompt=prompt,
                )
                reward = recalc_result.get("reward")
                delta_satisfaction = recalc_result.get("delta_satisfaction", 0.0)
                delta_awareness = recalc_result.get("delta_awareness", 0.0)
            elif score_type in {"multiturn", "singleturn"}:
                full_results = candidate.get("full_results") or {}
                results_dict = full_results.get(score_type) or {}
                reward = results_dict.get("reward", None)
                delta_satisfaction = results_dict.get("delta_satisfaction", 0.0)
                delta_awareness = results_dict.get("delta_awareness", 0.0)
            else:  # 'direct'
                reward = candidate.get("reward", None)
                delta_satisfaction = candidate.get("delta_satisfaction", 0.0)
                delta_awareness = candidate.get("delta_awareness", 0.0)

            row = {
                "artifact_id": artifact_id,
                "turn_id": str(turn_index),
                "assistant_index": assistant_index,
                "prompt": prompt,
                "completion": completion_text,
                "score": reward,
                "criteria_history": criteria_before_update if criteria_before_update is not None else [],
            }
            if conv_id is not None:
                row["conv_id"] = conv_id
            curr_rows.append(row)
            curr_result[str(assistant_index)] = reward
            curr_delta_satisfactions.append(delta_satisfaction)
            curr_delta_awareness.append(delta_awareness)

        if not curr_rows:
            n_skipped_turns += 1
            continue

        # Optionally filter out turns where all candidates have delta_awareness == 0
        if is_filter_zeros and all(da == 0 for da in curr_delta_awareness):
            n_skipped_turns += 1
            continue

        # Ensure candidate responses are distinct
        responses_distinct = True
        completions = [row["completion"] for row in curr_rows]
        for i, comp in enumerate(completions):
            if not is_distinct_response(comp, completions[i + 1 :]):
                responses_distinct = False
                break
        if not responses_distinct:
            n_skipped_turns += 1
            continue

        # Enforce a minimum score gap between candidates if requested
        if minimum_gap > 0.0:
            valid_scores = [score for score in curr_result.values() if score is not None]
            if len(valid_scores) >= 2:
                score_gap = max(valid_scores) - min(valid_scores)
                if score_gap < minimum_gap:
                    n_skipped_turns += 1
                    continue

        # Get the chosen row index which is the row with higher score
        chosen_row = max(curr_rows, key=lambda x: x["score"])
        full_conversation = chosen_row["prompt"] + [{"role": "assistant", "content": chosen_row["completion"]}]
        token_length = count_tokens_in_conversation(full_conversation)
        if token_length > max_token_length:
            n_skipped_turns += 1
            continue


        rows.extend(curr_rows)
        results.append(curr_result)

    return rows, results, n_skipped_turns


def extract_full_conversation(result_data: Dict[str, Any], system_prompt: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Extract conversations from synthesis data, creating one example per assistant turn.

    For a conversation with N assistant turns, this returns N separate conversation
    examples, each ending at a different assistant turn. This is the recommended
    format for training with train_on_what=LAST_ASSISTANT_MESSAGE.

    Args:
        result_data: A best_of_n.json (or legacy synthesis.json) payload as a dict
        system_prompt: Optional system prompt to prepend to messages

    Returns:
        List of dicts, each with 'messages' key containing a conversation ending
        at an assistant turn. Returns empty list if no valid conversations.
    """
    conversation_history = result_data.get(
        "conversation", result_data.get("conversation_history", [])
    )
    if not conversation_history:
        return []

    # Build base messages list
    all_messages = []
    if system_prompt:
        all_messages.append({"role": "system", "content": system_prompt})

    for msg in conversation_history:
        if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
            all_messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

    # Create one conversation example per assistant turn
    conversations = []
    for i, msg in enumerate(all_messages):
        if msg.get("role") == "assistant":
            # Include all messages up to and including this assistant turn
            conv_slice = all_messages[:i + 1]
            if len(conv_slice) > (1 if system_prompt else 0):  # Must have more than just system prompt
                conversations.append({"messages": conv_slice})

    return conversations


def process_all_runs(
    directory: str,
    score_type: str = "multiturn",
    end_index: Optional[int] = None,
    minimum_gap: float = 0.0,
    add_labels: bool = False,
    is_filter_zeros: bool = False,
    recalculate_rewards: bool = False,
    max_token_length: int = 4096,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], int, List[Dict[str, Any]]]:
    """
    Walk a synthesis results directory and aggregate per-turn candidate rows.

    Expected layout:
        <directory>/
            artifact_1/best_of_n.json
            artifact_2/best_of_n.json
            ...

    Args:
        minimum_gap: If > 0, require that the per-turn score difference between
            the highest and lowest non-None candidate scores is at least this
            value; otherwise the turn is skipped.
        recalculate_rewards: If True, recalculate rewards from stored criteria data.

    Returns:
        Tuple of (rows, results, total_skipped_turns, conversations)
        - rows: per-turn candidate data
        - results: per-turn score mappings
        - total_skipped_turns: number of skipped turns
        - conversations: list of full conversations in {"messages": [...]} format
    """
    all_rows: List[Dict[str, Any]] = []
    all_results: List[Dict[str, Any]] = []
    all_conversations: List[Dict[str, Any]] = []
    total_skipped_turns = 0

    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Synthesis directory not found: {directory}")

    for artifact_dir in sorted(base.iterdir()):
        if not artifact_dir.is_dir():
            continue

        artifact_id = artifact_dir.name
        # Pick a stable conv_id from the artifact directory name. We accept
        # either an integer suffix (``artifact_123``) or fall back to the
        # full directory name as a string id.
        conv_id = None
        if artifact_id.startswith("artifact_"):
            parts = artifact_id.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                conv_id = parts[1]
                conv_id_num = int(parts[1])
            else:
                conv_id_num = None
        else:
            # Try to interpret trailing digits, if any
            suffix = "".join(ch for ch in artifact_id if ch.isdigit())
            if suffix.isdigit():
                conv_id = suffix
                conv_id_num = int(suffix)
            else:
                conv_id_num = None

        if end_index is not None and conv_id_num is not None and conv_id_num > end_index:
            continue

        # Prefer the new ``best_of_n.json`` filename; fall back to legacy
        # ``synthesis.json`` so old experiment outputs still load.
        synth_file = artifact_dir / "best_of_n.json"
        if not synth_file.exists():
            synth_file = artifact_dir / "synthesis.json"
        if not synth_file.exists():
            continue

        result_data = _load_run_payload(synth_file)
        if not result_data:
            continue

        rows, results, n_skipped = process_single_run(
            result_data,
            artifact_id=artifact_id,
            conv_id=conv_id,
            minimum_gap=minimum_gap,
            score_type=score_type,
            add_labels=add_labels,
            is_filter_zeros=is_filter_zeros,
            recalculate_rewards=recalculate_rewards,
            max_token_length=max_token_length,
        )
        all_rows.extend(rows)
        all_results.extend(results)
        total_skipped_turns += n_skipped

        # Extract conversations (one per assistant turn)
        conversations = extract_full_conversation(result_data, system_prompt=SYSTEM_PROMPT)
        all_conversations.extend(conversations)

    return all_rows, all_results, total_skipped_turns, all_conversations


def summarize_results(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute summary statistics over per-turn assistant scores.
    """
    num_wins: Dict[str, int] = {}
    assistant_scores: Dict[str, List[float]] = {}
    decision_scores = {"winner": [], "loser": []}
    diffs: List[float] = []

    num_ties = 0
    for result in all_results:
        filtered_result = {k: v for k, v in result.items() if v is not None}
        if not filtered_result:
            continue

        winner_index = max(filtered_result, key=filtered_result.get)
        loser_index = min(filtered_result, key=filtered_result.get)
        winner_score = filtered_result[winner_index]
        loser_score = filtered_result[loser_index]

        for assistant_index, score in filtered_result.items():
            assistant_scores.setdefault(assistant_index, []).append(score)
            num_wins.setdefault(assistant_index, 0)

        # Only count a win if there's an actual difference in scores
        if winner_score > loser_score:
            decision_scores["winner"].append(winner_score)
            decision_scores["loser"].append(loser_score)
            diffs.append(winner_score - loser_score)
            num_wins[winner_index] += 1
        else:
            num_ties += 1

    print("Number of times each assistant wins:")
    total_decisions = len(all_results) if all_results else 1
    for assistant_index, num_win in num_wins.items():
        pct = (num_win / total_decisions) * 100
        print(f"\tAssistant {assistant_index}: {num_win} ({pct:.2f}%)")
    print(f"\tTies (equal scores): {num_ties} ({num_ties / total_decisions * 100:.2f}%)")

    print("Average + STDEV score for each assistant's turns:")
    for assistant_index, scores in assistant_scores.items():
        print(f"\tAssistant {assistant_index}: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    if decision_scores["winner"]:
        print("Average + STDEV score of winner and loser turns:")
        print(
            f"\tWinner: {np.mean(decision_scores['winner']):.2f} ± "
            f"{np.std(decision_scores['winner']):.2f}"
        )
        print(
            f"\tLoser: {np.mean(decision_scores['loser']):.2f} ± "
            f"{np.std(decision_scores['loser']):.2f}"
        )
    if diffs:
        print("Average + STDEV difference between winner and loser turns:")
        print(f"\t{np.mean(diffs):.2f} ± {np.std(diffs):.2f}")

    return {
        "num_wins": num_wins,
        "num_ties": num_ties,
        "assistant_scores": {k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in assistant_scores.items()},
    }


def print_turn_distribution(rows: List[Dict[str, Any]]) -> None:
    """
    Print the distribution of turn_id values across all rows.

    Args:
        rows: List of row dicts, each with 'turn_id' field
    """
    # Count occurrences of each turn_id across all rows
    turn_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        turn_id = row.get("turn_id", "")
        if turn_id:
            turn_counts[turn_id] += 1

    if not turn_counts:
        print("No turn distribution data available.")
        return

    # Convert turn_id to int for sorting, but keep original string for display
    total_rows = len(rows)

    print("\nDistribution of turns across all rows:")
    # Sort by turn_id as integer if possible, otherwise as string
    sorted_turns = sorted(turn_counts.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    for turn_id in sorted_turns:
        count = turn_counts[turn_id]
        pct = (count / total_rows) * 100
        print(f"\tTurn {turn_id}: {count} row(s) ({pct:.2f}%)")

    print(f"\nTotal rows: {total_rows}")
    print(f"Unique turn IDs: {len(turn_counts)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process synthesis (multiplex) results into a training/eval dataset.",
    )
    parser.add_argument(
        "--input_dir",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more directories containing <artifact_id>/best_of_n.json files. "
            "If multiple are provided, results are processed per-directory and then merged "
            "into a single dataset."
        ),
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Output path for the processed dataset.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="If set, shuffle the merged dataset once before saving.",
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=0,
        help="Random seed used when --shuffle is set.",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="Only process artifact folders with numeric IDs <= end_index (if parseable).",
    )
    parser.add_argument(
        "--save_format",
        choices=["hf", "json", "jsonl"],
        default="hf",
        help=(
            "Output format: 'hf' saves a Hugging Face dataset via save_to_disk, "
            "'json' saves a single JSON array, 'jsonl' saves JSON lines."
        ),
    )
    parser.add_argument(
        "--score_type",
        choices=["singleturn", "multiturn", "direct"],
        default="multiturn",
        help=(
            "Which score to use: 'multiturn'/'singleturn' from candidate['full_results'], "
            "or 'direct' from candidate['reward']."
        ),
    )
    parser.add_argument(
        "--minimum_gap",
        type=float,
        default=0.0,
        help=(
            "Minimum required difference between the highest and lowest non-None "
            "candidate scores within a turn. Turns with smaller gaps are skipped."
        ),
    )
    parser.add_argument(
        "--add_labels",
        action="store_true",
        default=False,
        help="If set, prepend an ACTION label with assistant index to each assistant response.",
    )
    parser.add_argument(
        "--is_filter_zeros",
        action="store_true",
        default=False,
        help="If set, filter out turns where all candidates have delta_awareness == 0.",
    )
    parser.add_argument(
        "--recalculate_rewards",
        action="store_true",
        default=False,
        help=(
            "If set, recalculate rewards from stored criteria data (updated_criteria_objs) "
            "instead of using pre-computed rewards. Uses the formula: "
            "reward = delta_awareness * 2 - token_penalty."
        ),
    )
    parser.add_argument(
        "--max_token_length",
        type=int,
        default=4096,
        help="Maximum token length for the conversation.",
    )
    args = parser.parse_args()

    input_dirs: List[str] = list(args.input_dir)

    merged_data: List[Dict[str, Any]] = []
    merged_results: List[Dict[str, Any]] = []
    merged_conversations: List[Dict[str, Any]] = []
    total_skipped_turns = 0

    for d in input_dirs:
        data, results, skipped_turns, conversations = process_all_runs(
            d,
            score_type=args.score_type,
            end_index=args.end_index,
            minimum_gap=args.minimum_gap,
            add_labels=args.add_labels,
            is_filter_zeros=args.is_filter_zeros,
            recalculate_rewards=args.recalculate_rewards,
            max_token_length=args.max_token_length,
        )

        # If merging multiple directories, retain provenance to avoid ambiguity when artifact
        # IDs overlap across runs (e.g., many runs contain `artifact_1`, `artifact_2`, ...).
        if len(input_dirs) > 1:
            for row in data:
                row["source_dir"] = d

        merged_data.extend(data)
        merged_results.extend(results)
        merged_conversations.extend(conversations)
        total_skipped_turns += skipped_turns

    if args.shuffle and merged_data:
        rng = random.Random(args.shuffle_seed)
        rng.shuffle(merged_data)
        if merged_conversations:
            rng.shuffle(merged_conversations)

    total_rows = len(merged_data)
    denom = total_rows + total_skipped_turns if (total_rows + total_skipped_turns) > 0 else 1
    if args.recalculate_rewards:
        print("Results aggregated using RECALCULATED rewards (from stored criteria data).\n")
    else:
        print(f"Results aggregated using {args.score_type} scores.\n")
    print(f"Total skipped turns: {total_skipped_turns} ({total_skipped_turns / denom * 100:.2f}%)")
    summarize_results(merged_results)
    print_turn_distribution(merged_data)

    print(f"Saving {len(merged_data)} rows to {args.output} in {args.save_format} format")
    save_dataset(merged_data, args.output, args.save_format)

    # Save full conversations as JSONL
    if merged_conversations:
        conversations_path = Path(args.output).parent / f"{Path(args.output).stem}_conversations.jsonl"
        save_conversations_jsonl(merged_conversations, str(conversations_path))
        print(f"Saved {len(merged_conversations)} full conversations to {conversations_path}")

    # Save a random example row for inspection
    if merged_data:
        example_row = random.choice(merged_data)
        example_path = Path(args.output).parent / f"{Path(args.output).stem}_example.json"
        with example_path.open("w", encoding="utf-8") as f:
            json.dump(example_row, f, indent=2, ensure_ascii=False)
        print(f"Saved random example row to {example_path}")


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()


