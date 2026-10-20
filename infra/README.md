# Infrastructure

Terraform for the GCP project `saas-458416`. It creates APIs, Artifact Registry, two service
accounts, Workload Identity Federation for GitHub Actions, one Cloud Run service and one
Cloud Run job.

**Who runs it:** a person with project Owner on `saas-458416`. The apply changes IAM and
enables APIs. The bucket `gs://nba-data` must already exist. Terraform does not manage it.

## Steps

1. Sign in with the Owner account: `gcloud auth application-default login`
2. Go to the folder: `cd infra`
3. Run `terraform init`
4. Run `terraform plan`
5. Read the plan. Confirm it creates only new resources.
6. Run `terraform apply`
7. Run `terraform output` and copy `wif_provider` and `deploy_sa_email`
8. In GitHub, open Settings, Secrets and variables, Actions, Variables. Add `WIF_PROVIDER` and `DEPLOY_SA` with those values.

## Bootstrap image

No image exists in Artifact Registry before the first deploy. The variable `image` defaults to
Google's public hello image (`us-docker.pkg.dev/cloudrun/container/hello`), so the first apply
succeeds. The GitHub deploy workflow then replaces the image on the service and the job.
Terraform ignores the image field, so a later `terraform apply` does not revert it.

The hello image ignores the job arg `replay`. Do not run the job until the deploy workflow has
pushed the real image.
