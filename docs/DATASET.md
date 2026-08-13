# Dataset

## Source

CloudTrace-MVAD uses the Kaggle distribution **AWS Cloudtrails Dataset from flaws.cloud** and the file `nineteenFeaturesDf.csv`.

- Original public release description: Scott Piper, *Public dataset of CloudTrail logs from flaws.cloud*, Summit Route, 9 October 2020.
- Kaggle distribution: `nobukim/aws-cloudtrails-dataset-from-flaws-cloud`.

The original release documents 1,939,207 CloudTrail events from 12 February 2017 through 7 October 2020, 9,402 source IP values, 8,811 user agents, and 1,242 attempted AWS APIs.

## Curation rule

The paper run loads the detailed CSV directly, parses timestamps in UTC, removes rows with invalid timestamps, and removes repeated `eventID` values before any session construction or feature fitting. The archived run removed 713,423 repeated identifiers and retained 1,225,784 unique events.

## Label status

The dataset does not provide authoritative event-level benign/malicious ground truth. The repository therefore does not create supervised attack labels from the source records. Controlled perturbations are separate evaluation constructs applied after the chronological split.

## Redistribution

The dataset is not bundled here. Obtain it from the host and comply with the host and source terms.
