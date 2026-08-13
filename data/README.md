# Dataset placement

Download **AWS Cloudtrails Dataset from flaws.cloud** from Kaggle and place only the following file here:

```text
data/nineteenFeaturesDf.csv
```

Dataset page:

https://www.kaggle.com/datasets/nobukim/aws-cloudtrails-dataset-from-flaws-cloud

The study intentionally excludes `dec12_18features.csv`.

The loader resolves common flattened CloudTrail column-name variants and reports the detected schema before processing. Core required information is event time, event identifier, event name, event source, region, source IP, user agent, error code, and at least one usable identity field.
