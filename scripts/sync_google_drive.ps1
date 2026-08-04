param(
    [ValidateSet("Download", "Upload", "UploadExpanded")]
    [string]$Action,
    [string]$RemoteName = "astroseg-drive",
    [string]$Workspace = ".astroseg_runtime"
)

$ErrorActionPreference = "Stop"
$RootFolderId = "15FrdmbZGEWyB2mBgGv2hVpIlkcGQy6tE"
$WorkspacePath = [IO.Path]::GetFullPath((Join-Path (Get-Location) $Workspace))
$RepositoryPath = [IO.Path]::GetFullPath((Get-Location).Path)

if (-not $WorkspacePath.StartsWith($RepositoryPath + [IO.Path]::DirectorySeparatorChar)) {
    throw "Workspace must remain inside the repository working directory."
}
$RcloneCommand = Get-Command rclone -ErrorAction SilentlyContinue
if ($RcloneCommand) {
    $Rclone = $RcloneCommand.Source
} else {
    # A newly installed WinGet alias is not visible to an already-open terminal.
    $PackageRoot = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Packages"
    $Rclone = Get-ChildItem -Path $PackageRoot -Filter rclone.exe -Recurse `
        -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $Rclone) {
    throw "rclone is not installed. Install it, then run: rclone config"
}

if ($Action -eq "Download") {
    $TrainingImages = Join-Path $WorkspacePath "dataset\training_images"
    $TrainingMasks = Join-Path $WorkspacePath "dataset\training_masks"
    $TestImages = Join-Path $WorkspacePath "dataset\test_images"
    foreach ($Directory in ($TrainingImages, $TrainingMasks, $TestImages)) {
        New-Item -ItemType Directory -Path $Directory -Force | Out-Null
    }
    & $Rclone copy "${RemoteName}:Astrocytes Training Photos" $TrainingImages `
        --drive-root-folder-id $RootFolderId --progress
    if ($LASTEXITCODE -ne 0) { throw "Training-image download failed." }
    & $Rclone copy "${RemoteName}:Astrocytes Training Masks/Astrocytes Final Masks" $TrainingMasks `
        --drive-root-folder-id $RootFolderId --progress
    if ($LASTEXITCODE -ne 0) { throw "Training-mask download failed." }
    & $Rclone copy "${RemoteName}:Astrocytes Morphology Photos" $TestImages `
        --drive-root-folder-id $RootFolderId --progress
    if ($LASTEXITCODE -ne 0) { throw "Test-image download failed." }
    Write-Host "Drive dataset downloaded to $WorkspacePath"
    exit 0
}

$OutputPath = Join-Path $WorkspacePath "outputs"
if (-not (Test-Path -LiteralPath $OutputPath -PathType Container)) {
    throw "No runtime output directory exists at $OutputPath"
}
if ($Action -eq "UploadExpanded") {
    & $Rclone copy $OutputPath "${RemoteName}:Astroseg Outputs" `
        --drive-root-folder-id $RootFolderId --transfers 16 --checkers 32 `
        --fast-list --progress
    if ($LASTEXITCODE -ne 0) { throw "Expanded output upload failed." }
    Write-Host "Expanded runtime-output tree uploaded to Astroseg Outputs."
    exit 0
}
$TarCommand = Get-Command tar -ErrorAction SilentlyContinue
if (-not $TarCommand) {
    throw "tar is required to create the compressed output archive."
}
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$ArchiveName = "astroseg_outputs_${Timestamp}.tar.zst"
$ArchivePath = Join-Path $WorkspacePath $ArchiveName
& $TarCommand.Source --zstd -cf $ArchivePath -C $WorkspacePath outputs
if ($LASTEXITCODE -ne 0) { throw "Output archive creation failed." }
$UploadSucceeded = $false
try {
    & $Rclone copyto $ArchivePath "${RemoteName}:Astroseg Outputs/$ArchiveName" `
        --drive-root-folder-id $RootFolderId --progress
    if ($LASTEXITCODE -ne 0) { throw "Output archive upload failed." }
    $UploadSucceeded = $true
} finally {
    if ($UploadSucceeded -and (Test-Path -LiteralPath $ArchivePath -PathType Leaf)) {
        Remove-Item -LiteralPath $ArchivePath
    }
}
Write-Host "Complete runtime-output archive uploaded as $ArchiveName"
