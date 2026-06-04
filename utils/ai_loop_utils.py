from datetime import datetime, time, timedelta


def parse_deadline(deadline_text, now=None):
    now = now or datetime.now()
    deadline_text = (deadline_text or "").strip()
    if not deadline_text:
        raise ValueError("deadline_text must be provided")

    lowered = deadline_text.lower()
    if lowered in {"tonight_24", "midnight", "tonight"}:
        next_day = now.date() + timedelta(days=1)
        return datetime.combine(next_day, time.min).replace(tzinfo=now.tzinfo)

    if "t" in deadline_text:
        return datetime.fromisoformat(deadline_text)

    if len(deadline_text) == 5 and ":" in deadline_text:
        hour_text, minute_text = deadline_text.split(":")
        if hour_text == "24" and minute_text == "00":
            next_day = now.date() + timedelta(days=1)
            return datetime.combine(next_day, time.min).replace(tzinfo=now.tzinfo)
        target = now.replace(
            hour=int(hour_text),
            minute=int(minute_text),
            second=0,
            microsecond=0,
        )
        if target <= now:
            target = target + timedelta(days=1)
        return target

    raise ValueError(f"Unsupported deadline format: {deadline_text}")


def should_continue(now, deadline, iteration, max_iterations=None):
    if max_iterations is not None and iteration >= int(max_iterations):
        return False
    return now < deadline


def build_loop_summary(started_at, deadline, iteration_count, success_count, failure_count):
    return {
        "started_at": started_at.isoformat(),
        "deadline": deadline.isoformat(),
        "iteration_count": int(iteration_count),
        "success_count": int(success_count),
        "failure_count": int(failure_count),
    }
