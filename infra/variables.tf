variable "project_id" {
  type    = string
  default = "saas-458416"
}

variable "region" {
  type    = string
  default = "us-central1"
}

# Placeholder so the first apply works before any image exists.
# CI replaces the image on every deploy; Terraform ignores that change.
variable "image" {
  type    = string
  default = "us-docker.pkg.dev/cloudrun/container/hello"
}
