terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=2.64.0"
    }
  }
  backend "remote" {
    # The name of your Terraform Cloud organization.
    organization = "pbazure"

    # The name of the Terraform Cloud workspace to store Terraform state files in.
    workspaces {
      name = "recommender-systems"
    }
  }
}


variable "subscription_id" {
  type = string
}
variable "client_id" {
  type = string
}
variable "client_secret" {
  type = string
}
variable "tenant_id" {
  type = string
}

# Configure the Microsoft Azure Provider
provider "azurerm" {
  features {}

  subscription_id = var.subscription_id
  client_id       = var.client_id
  client_secret   = var.client_secret
  tenant_id       = var.tenant_id
}

data "azurerm_client_config" "current" {}

resource "azurerm_resource_group" "aml" {
  name     = "azure-ml"
  location = "West US"
}

resource "azurerm_application_insights" "aml" {
  name                = "recommender-systems-ai"
  location            = azurerm_resource_group.aml.location
  resource_group_name = azurerm_resource_group.aml.name
  application_type    = "web"
}

resource "azurerm_key_vault" "aml" {
  name                     = "recommendersystemspb"
  location                 = azurerm_resource_group.aml.location
  resource_group_name      = azurerm_resource_group.aml.name
  tenant_id                = data.azurerm_client_config.current.tenant_id
  sku_name                 = "standard"
  purge_protection_enabled = true
}

#storage information
resource "azurerm_storage_account" "aml" {
  name                     = "recommendersystemsbucket"
  location                 = azurerm_resource_group.aml.location
  resource_group_name      = azurerm_resource_group.aml.name
  account_tier             = "Standard"
  account_replication_type = "GRS"
}
resource "azurerm_storage_container" "aml" {
  name                  = "ecommData"
  storage_account_name  = azurerm_storage_account.aml.name
  container_access_type = "private"
}

resource "azurerm_container_registry" "aml" {
  name                     = "recommendsysacr"
  resource_group_name      = azurerm_resource_group.aml.name
  location                 = azurerm_resource_group.aml.location
  sku                      = "Basic"
  admin_enabled            = true
}


resource "azurerm_machine_learning_workspace" "aml" {
  name                    = "recommend-pb-workspace"
  location                = azurerm_resource_group.aml.location
  resource_group_name     = azurerm_resource_group.aml.name
  application_insights_id = azurerm_application_insights.aml.id
  key_vault_id            = azurerm_key_vault.aml.id
  storage_account_id      = azurerm_storage_account.aml.id
  container_registry_id   = azurerm_container_registry.aml.id

  identity {
    type = "SystemAssigned"
  }
}