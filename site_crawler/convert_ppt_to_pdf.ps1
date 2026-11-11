param(
    [Parameter(Mandatory = $true)][string]$InputPath,
    [Parameter(Mandatory = $true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'
$powerPoint = $null
$presentation = $null
try {
    $powerPoint = New-Object -ComObject PowerPoint.Application
    $powerPoint.Visible = 0
    $powerPoint.DisplayAlerts = 1
    # msoAutomationSecurityForceDisable: never run embedded macros.
    try { $powerPoint.AutomationSecurity = 3 } catch { }
    # Open read-only and without a window; the original file remains untouched.
    $presentation = $powerPoint.Presentations.Open($InputPath, $true, $false, $false)
    # ppFixedFormatTypePDF = 2
    $presentation.ExportAsFixedFormat($OutputPath, 2)
    if (-not (Test-Path -LiteralPath $OutputPath -PathType Leaf)) {
        throw 'PowerPoint returned without creating the output PDF.'
    }
}
finally {
    if ($presentation -ne $null) { $presentation.Close() }
    if ($powerPoint -ne $null) { $powerPoint.Quit() }
    if ($presentation -ne $null) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($presentation) }
    if ($powerPoint -ne $null) { [void][Runtime.InteropServices.Marshal]::FinalReleaseComObject($powerPoint) }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}
