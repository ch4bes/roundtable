import json
from pathlib import Path
from datetime import datetime
from .session import Session
import aiofiles


class Exporter:
    @staticmethod
    def _format_matrix_table(model_names, matrix):
        n = len(model_names)
        if n == 0:
            return ""
        
        # Compute each data column's width
        col_widths = [len(model_names[j]) for j in range(n)]
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    val = f"{matrix[i][j]:.2f}"
                    if len(val) > col_widths[j]:
                        col_widths[j] = len(val)
        
        # Row label width = col 0's width (covers labels + first col data)
        row_width = col_widths[0]
        
        # Header: empty row_label cell + data columns
        # Use col_widths[0] as the empty label cell width (same as row labels)
        parts = [" " * row_width]
        for j in range(n):
            parts.append(model_names[j].rjust(col_widths[j]))
        lines = ["  ".join(parts)]
        
        # Separator row: spaces matching header's empty label + dashes for data cols
        sep_parts = [" " * row_width]
        for j in range(n):
            sep_parts.append("-" * col_widths[j])
        lines.append("  ".join(sep_parts))
        
        # Data rows
        for i in range(n):
            row_parts = [model_names[i].rjust(row_width)]
            for j in range(n):
                if i == j:
                    row_parts.append(" " * col_widths[j])
                else:
                    row_parts.append(f"{matrix[i][j]:>{col_widths[j]}.2f}")
            lines.append("  ".join(row_parts))
        
        return "\n".join(lines)

    @staticmethod
    def _format_matrix_md_table(model_names, matrix):
        n = len(model_names)
        if n == 0:
            return ""
        
        header = "| Model |" + " | ".join(name for name in model_names) + " |"
        separator = (
            "| --- |"
            + "".join(" | " + "-" * len(name) for name in model_names)
            + " |"
        )
        
        rows = []
        for i in range(n):
            cells = [f" **{model_names[i]}** "]
            for j in range(n):
                cells.append(f" {matrix[i][j]:.2f} ")
            rows.append("| " + " | ".join(cells) + "|")
        
        return header + "\n" + separator + "\n" + "\n".join(rows)

    @staticmethod
    async def export_markdown(session: Session, output_path: str | Path) -> Path:
        output_path = Path(output_path)

        lines = [
            "# LLM Roundtable Discussion",
            "",
            f"**Session ID:** {session.id}",
            f"**Prompt:** {session.prompt}",
            f"**Started:** {session.created_at}",
            f"**Status:** {session.status}",
            f"**Rounds:** {session.completed_rounds}",
            f"**Consensus:** {'Yes' if session.consensus_reached else 'No'}",
            "" if session.consensus_reached else "",
        ]

        if session.consensus_round:
            lines.append(f"**Consensus reached in round:** {session.consensus_round}")
            lines.append("")

        lines.extend(["---", ""])

        for round_num in range(1, session.completed_rounds + 1):
            round_responses = session.get_round_responses(round_num)
            round_summary = next(
                (s for s in session.summaries if s.round == round_num),
                None,
            )
            attributed = session.get_attributed_summary(round_num)

            lines.append(f"## Round {round_num}")
            lines.append("")

            for resp in round_responses:
                lines.append(f"### {resp.model}")
                lines.append("")
                if resp.response_time_s is not None:
                    lines.append(f"*Response time: {resp.response_time_s:.2f}s*")
                    lines.append("")
                lines.append(resp.content)
                lines.append("")

            if attributed:
                lines.append("### Attributed Analysis")
                lines.append("")
                for model, points in attributed.individual_summaries.items():
                    lines.append(f"**{model}**")
                    for point in points:
                        lines.append(f"- {point}")
                    lines.append("")
                lines.append(f"**Agreement Analysis:** {attributed.agreement_analysis}")
                lines.append("")
                lines.append(
                    f"**Consensus Assessment:** {attributed.consensus_assessment} (Confidence: {attributed.confidence})"
                )
                lines.append("")

            if round_summary:
                lines.append("### Summary")
                lines.append("")
                lines.append(round_summary.summary)
                lines.append("")

            sim = session.get_similarity_matrix(round_num)
            if sim:
                lines.append("### Similarity Matrix")
                lines.append("")
                table = Exporter._format_matrix_md_table(sim["model_names"], sim["matrix"])
                lines.append(table)
                n = len(sim["model_names"])
                matrix = sim["matrix"]
                threshold = session.config_snapshot.get("discussion", {}).get("consensus_threshold", 0.75)
                agreeing_pairs = sum(
                    1 for i in range(n) for j in range(i + 1, n) if matrix[i][j] >= threshold
                )
                total_pairs = n * (n - 1) // 2 if n > 1 else 0
                agreement_pct = (agreeing_pairs / total_pairs * 100) if total_pairs > 0 else 0
                lines.append("")
                lines.append(f"**Agreement:** {agreeing_pairs}/{total_pairs} pairs above {threshold} ({agreement_pct:.1f}%)")
                lines.append("")

            lines.append("---")
            lines.append("")

        if session.final_review:
            lines.append("## Final Review")
            lines.append("")
            lines.append(session.final_review)
            lines.append("")
            lines.append("---")
            lines.append("")

        if session.status == "running":
            lines.append("*Discussion in progress...*")
        elif session.status == "stopped":
            lines.append("*Discussion stopped by user.*")
        elif session.consensus_reached:
            lines.append("**Consensus reached!**")
        else:
            lines.append("*Discussion completed without consensus.*")

        content = "\n".join(lines)
        async with aiofiles.open(output_path, "w") as f:
            await f.write(content)

        return output_path

    @staticmethod
    async def export_json(session: Session, output_path: str | Path) -> Path:
        output_path = Path(output_path)
        data = session.to_dict()
        data["exported_at"] = datetime.now().isoformat()

        async with aiofiles.open(output_path, "w") as f:
            await f.write(json.dumps(data, indent=2))

        return output_path

    @staticmethod
    async def export(
        session: Session,
        output_path: str | Path,
        format: str = "md",
    ) -> Path:
        if format == "json":
            return await Exporter.export_json(session, output_path)
        else:
            return await Exporter.export_markdown(session, output_path)
