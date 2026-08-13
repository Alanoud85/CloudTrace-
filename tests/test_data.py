import pandas as pd

from cloudtrace_mvad.data import detect_schema, load_cloudtrail_csv, sessionize, chronological_split


def test_sessionization_and_split(tmp_path):
    df = pd.DataFrame({
        "eventTime": ["2020-01-01T00:00:00Z", "2020-01-01T00:05:00Z", "2020-01-01T01:00:00Z", "2020-01-02T00:00:00Z"],
        "eventID": ["a", "b", "c", "d"],
        "eventName": ["A", "B", "C", "D"],
        "eventSource": ["s", "s", "s", "s"],
        "awsRegion": ["r", "r", "r", "r"],
        "sourceIPAddress": ["1", "1", "1", "2"],
        "userAgent": ["u", "u", "u", "u"],
        "errorCode": [None, None, "Denied", None],
        "principalId": ["p", "p", "p", "q"],
    })
    path = tmp_path / "x.csv"; df.to_csv(path, index=False)
    events, schema, audit = load_cloudtrail_csv(path)
    events, sessions = sessionize(events, 30)
    assert audit["curated_rows"] == 4
    assert len(sessions) == 3
    split = chronological_split(sessions, 2/3, 0.0)
    assert (split["split"] == "train").sum() == 2
