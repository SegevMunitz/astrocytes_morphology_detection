param(
    [string]$RemoteName = "astroseg-drive",
    [int]$Fold = 0,
    [string]$Annotator = "",
    [switch]$SkipDownload,
    [switch]$SkipUpload
)

$ErrorActionPreference = "Stop"
$RepositoryPath = [IO.Path]::GetFullPath((Get-Location).Path)
$Workspace = ".astroseg_runtime"
$WorkspacePath = [IO.Path]::GetFullPath((Join-Path $RepositoryPath $Workspace))
if (-not $WorkspacePath.StartsWith($RepositoryPath + [IO.Path]::DirectorySeparatorChar)) {
    throw "Workspace must remain inside the repository working directory."
}

$LocalPython = Join-Path $RepositoryPath ".python311\python.exe"
if (Test-Path -LiteralPath $LocalPython -PathType Leaf) {
    $Python = $LocalPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = "python"
} else {
    throw "Python was not found. Create the environment described in README.md first."
}

$Runtime = $Workspace.Replace("\", "/").TrimEnd("/")
$Manifest = "$Runtime/outputs/metadata/manifest.csv"
$InstanceManifest = "$Runtime/outputs/metadata/manifest_instances.csv"
$Masks = "$Runtime/dataset/training_masks"
$Outputs = "$Runtime/outputs"

if (-not $SkipDownload) {
    & powershell -ExecutionPolicy Bypass -File scripts/sync_google_drive.ps1 `
        -Action Download -RemoteName $RemoteName -Workspace $Workspace
    if ($LASTEXITCODE -ne 0) { throw "Google Drive download failed." }
}

& $Python scripts/build_drive_manifest.py `
    --training-images "$Runtime/dataset/training_images" `
    --test-images "$Runtime/dataset/test_images" --output $Manifest
if ($LASTEXITCODE -ne 0) { throw "Manifest creation failed." }

& $Python scripts/prepare_dataset.py --manifest $Manifest `
    --output-dir "$Outputs/interim"
if ($LASTEXITCODE -ne 0) { throw "Channel and nucleus preparation failed." }

$ImportArguments = @(
    "scripts/import_instance_annotations.py",
    "--manifest", $Manifest,
    "--cellpose-dir", $Masks,
    "--output-dir", "$Outputs/annotations",
    "--output-manifest", $InstanceManifest,
    "--status", "seed",
    "--overwrite"
)
if ($Annotator) { $ImportArguments += @("--annotator", $Annotator) }
& $Python @ImportArguments
if ($LASTEXITCODE -ne 0) { throw "Training-mask import failed." }

& $Python scripts/train_instances.py --config configs/train_instances_drive.yaml --fold $Fold
if ($LASTEXITCODE -ne 0) { throw "Instance-model training failed." }

$Checkpoint = "$Outputs/checkpoints/astrocyte_instances/best.pt"
& $Python scripts/predict_astrocyte_instances.py `
    --config configs/train_instances_drive.yaml --checkpoint $Checkpoint --split test `
    --output-dir "$Outputs/instance_predictions" `
    --output-manifest "$Outputs/instance_predictions/manifest.csv" --overwrite
if ($LASTEXITCODE -ne 0) { throw "Test-image prediction failed." }

if (-not $SkipUpload) {
    & powershell -ExecutionPolicy Bypass -File scripts/sync_google_drive.ps1 `
        -Action Upload -RemoteName $RemoteName -Workspace $Workspace
    if ($LASTEXITCODE -ne 0) { throw "Google Drive output upload failed." }
}

Write-Host "Pipeline completed. Runtime files: $Outputs"
