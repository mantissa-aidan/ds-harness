"""Reports use computed evidence as the authoritative source."""

from ds_harness.config import Task


def render_report(task: Task, state: dict, selection: dict, preparation: dict,
                  results: dict, evaluation: dict, interpretation: dict | None) -> str:
    lines = [f"# {task.name}", "", f"**Run status:** {state['status']}", "",
             "## 1. Business understanding", "", task.objective, "",
             f"Decision: {task.decision}", "", f"Reference date (UTC): {task.as_of}", "",
             "Business success criteria (require owner validation):", ""]
    lines += [f"- {criterion}" for criterion in task.success_criteria]
    lines += ["", "## 2. Data understanding", "",
              "Candidates are evaluated in configured order; the first contract-valid candidate is selected.", ""]
    for dataset, record in selection.items():
        lines.append(f"- **{dataset}**: selected candidate {record.get('selected', 'none')}; "
                     f"{len(record.get('attempts', []))} candidate(s) examined.")
        for attempt in record.get("attempts", []):
            failed = [c["check"] for c in attempt.get("quality", {}).get("checks", []) if not c["passed"]]
            detail = attempt.get("error") or (", ".join(failed) if failed else "contract passed")
            lines.append(f"  - Candidate {attempt['candidate']} ({attempt['source']}): {detail}")
    lines += ["", "Full evidence: [data_understanding.json](data_understanding.json).", "",
              "## 3. Data preparation", ""]
    for dataset, record in preparation.get("datasets", {}).items():
        lines.append(f"- **{dataset}**: {record['input_rows']} → {record['output_rows']} rows; "
                     "all transformations are recorded in data_preparation.json.")
    checks = preparation.get("integrity", {}).get("checks", [])
    lines.append(f"\nIntegrity checks: {sum(bool(c['passed']) for c in checks)}/{len(checks)} passed.")
    lines += ["", "## 4. Modeling and analysis", ""]
    if not results:
        lines.append("No analytical conclusions were produced.")
    for analysis_id, result in results.items():
        lines += [f"### {analysis_id}", "", f"Kind: {result['kind']}. "
                  f"Evaluation: {'passed' if result.get('passed') else 'inconclusive / failed'}.", ""]
        if "error" in result:
            lines += [result["error"], ""]
        if result.get("metrics"):
            lines += ["| Computed metric | Value |", "| --- | ---: |"]
            for metric, value in result["metrics"].items():
                display = f"{value:.6g}" if isinstance(value, float) else str(value)
                lines.append(f"| {metric} | {display} |")
            lines += ["", f"[Result table](analyses/{analysis_id}.csv)", ""]
        lines += [f"- {limitation}" for limitation in result.get("limitations", [])]
        lines.append("")
    lines += ["## 5. Evaluation", "",
              f"Technical acceptance: **{'passed' if evaluation.get('passed') else 'not established'}**.", "",
              "A technical pass does not establish business impact or authorize a campaign/model deployment.", ""]
    lines += [f"- {issue}" for issue in evaluation.get("issues", [])]
    if interpretation:
        lines += ["", "### LLM interpretation (review against computed evidence)", ""]
        for observation in interpretation["observations"]:
            lines += [f"- {observation['interpretation']} "
                      f"Evidence: {', '.join(observation['evidence_ids'])}. "
                      f"Suggested action: {observation['suggested_action']}"]
        lines += [f"- Caveat: {c}" for c in interpretation["caveats"]]
    lines += ["", "## 6. Deployment / handoff", "",
              "This run produces a local decision-support package. Database and source-system writes "
              "are unavailable. No model, campaign, or business-system change has been deployed.", "",
              "Use handoff.json for monitoring, remediation, ownership and follow-up experiments. "
              "Re-run with a new directory after changing data definitions or acceptance criteria.", "",
              "Source snapshots and reports may contain confidential data. They stay in this local "
              "run directory and are excluded from Git by the supplied .gitignore.", ""]
    return "\n".join(lines)
