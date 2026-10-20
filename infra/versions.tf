terraform {
  required_version = ">= 1.5"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  backend "gcs" {
    bucket = "nba-data"
    prefix = "tfstate"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}
