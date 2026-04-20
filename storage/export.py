import json
from pathlib import Path
from datetime import datetime
from .session import Session
import aiofiles


class Exporter:
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
