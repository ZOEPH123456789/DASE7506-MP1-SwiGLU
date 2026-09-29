param(
    [Parameter(Mandatory=$true)][string]$PythonPath,
    [Parameter(Mandatory=$true)][string]$WorkingDirectory,
    [Parameter(Mandatory=$true)][string]$CheckpointPath,
    [Parameter(Mandatory=$true)][string]$OutputPath
)

$arguments = @(
    'evaluate.py', '--checkpoint', $CheckpointPath,
    '--device', 'cpu', '--precision', 'fp32', '--threads', '4',
    '--split', 'test', '--output', $OutputPath
)

$process = Start-Process -FilePath $PythonPath -ArgumentList $arguments `
    -WorkingDirectory $WorkingDirectory -WindowStyle Hidden -PassThru
$peakBytes = 0L

function Get-DescendantProcessIds([int]$ParentId) {
    $result = @()
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        $result += [int]$child.ProcessId
        $result += Get-DescendantProcessIds -ParentId ([int]$child.ProcessId)
    }
    return $result
}

do {
    $ids = @([int]$process.Id) + @(Get-DescendantProcessIds -ParentId ([int]$process.Id))
    $sampleBytes = 0L
    foreach ($id in ($ids | Select-Object -Unique)) {
        try {
            $sampleBytes += [int64](Get-Process -Id $id -ErrorAction Stop).WorkingSet64
        } catch {}
    }
    if ($sampleBytes -gt $peakBytes) { $peakBytes = $sampleBytes }
    Start-Sleep -Milliseconds 100
    $process.Refresh()
} while (-not $process.HasExited)

[pscustomobject]@{
    peak_working_set_bytes = $peakBytes
    peak_working_set_gib = [math]::Round($peakBytes / 1GB, 4)
    process_exit_code = $process.ExitCode
    output_path = $OutputPath
} | ConvertTo-Json
