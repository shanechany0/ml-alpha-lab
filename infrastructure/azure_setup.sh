#!/usr/bin/env bash
# Azure infrastructure setup for ML Alpha Lab
set -euo pipefail

# ── Variables ────────────────────────────────────────────────────────────────
RESOURCE_GROUP="ml-alpha-lab-rg"
WORKSPACE_NAME="ml-alpha-lab-ws"
LOCATION="eastus"
STORAGE_ACCOUNT="mlalphalabstorage"
CONTAINER_NAME="ml-alpha-lab"
ACR_NAME="mlalphalabacr"
KEYVAULT_NAME="ml-alpha-lab-kv"
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID:-}"

if [[ -z "$SUBSCRIPTION_ID" ]]; then
    echo "ERROR: AZURE_SUBSCRIPTION_ID environment variable is not set."
    exit 1
fi

echo "==> Setting active subscription: $SUBSCRIPTION_ID"
az account set --subscription "$SUBSCRIPTION_ID"

# ── Resource Group ────────────────────────────────────────────────────────────
echo "==> Creating resource group: $RESOURCE_GROUP"
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# ── Azure ML Workspace ────────────────────────────────────────────────────────
echo "==> Creating Azure ML workspace: $WORKSPACE_NAME"
az ml workspace create \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# ── CPU Compute Cluster ───────────────────────────────────────────────────────
echo "==> Creating CPU compute cluster"
az ml compute create \
    --name "cpu-cluster" \
    --type AmlCompute \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --size Standard_DS3_v2 \
    --min-instances 0 \
    --max-instances 4 \
    --idle-time-before-scale-down 300 \
    --output table

# ── GPU Compute Cluster ───────────────────────────────────────────────────────
echo "==> Creating GPU compute cluster"
az ml compute create \
    --name "gpu-cluster" \
    --type AmlCompute \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$WORKSPACE_NAME" \
    --size Standard_NC6 \
    --min-instances 0 \
    --max-instances 2 \
    --idle-time-before-scale-down 300 \
    --output table

# ── Storage Account ───────────────────────────────────────────────────────────
echo "==> Creating storage account: $STORAGE_ACCOUNT"
az storage account create \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output table

STORAGE_KEY=$(az storage account keys list \
    --account-name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --query "[0].value" \
    --output tsv)

echo "==> Creating blob container: $CONTAINER_NAME"
az storage container create \
    --name "$CONTAINER_NAME" \
    --account-name "$STORAGE_ACCOUNT" \
    --account-key "$STORAGE_KEY" \
    --output table

# ── Azure Container Registry ──────────────────────────────────────────────────
echo "==> Creating Azure Container Registry: $ACR_NAME"
az acr create \
    --name "$ACR_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --sku Basic \
    --output table

# ── Key Vault ─────────────────────────────────────────────────────────────────
echo "==> Creating Key Vault: $KEYVAULT_NAME"
az keyvault create \
    --name "$KEYVAULT_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# ── Output Summary ────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " Azure ML Alpha Lab infrastructure setup complete"
echo "============================================================"
echo " Resource Group : $RESOURCE_GROUP"
echo " ML Workspace   : $WORKSPACE_NAME"
echo " Location       : $LOCATION"
echo " Storage Acct   : $STORAGE_ACCOUNT"
echo " ACR            : $ACR_NAME"
echo " Key Vault      : $KEYVAULT_NAME"
echo ""

WORKSPACE_JSON=$(az ml workspace show \
    --name "$WORKSPACE_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --output json)

echo "Workspace details:"
echo "$WORKSPACE_JSON" | python3 -c "
import json, sys
ws = json.load(sys.stdin)
print(f'  ID       : {ws[\"id\"]}')
print(f'  Location : {ws[\"location\"]}')
print(f'  MLflow   : {ws.get(\"mlflow_tracking_uri\", \"N/A\")}')
"

STORAGE_CONN=$(az storage account show-connection-string \
    --name "$STORAGE_ACCOUNT" \
    --resource-group "$RESOURCE_GROUP" \
    --query "connectionString" \
    --output tsv)

echo ""
echo " Storage connection string:"
echo "  $STORAGE_CONN"
echo "============================================================"
