output "wif_provider" {
  description = "Value for the GitHub repository variable WIF_PROVIDER."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "deploy_sa_email" {
  description = "Value for the GitHub repository variable DEPLOY_SA."
  value       = google_service_account.deploy.email
}

output "ar_repo_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.nbaxp.repository_id}"
}
