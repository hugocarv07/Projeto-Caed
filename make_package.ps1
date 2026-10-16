$out = "caed_codigo.zip"
$tmp = Join-Path $env:TEMP "caed_pkg"

if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
if (Test-Path $out) { Remove-Item $out -Force }
New-Item -ItemType Directory -Path $tmp | Out-Null

$itens = @("caed", "scripts", "templates", "tests", "app.py", "README.md", "RELATORIO.md", "requirements.txt")
foreach ($i in $itens) {
    if (Test-Path $i) {
        Copy-Item $i -Destination $tmp -Recurse
    } else {
        Write-Warning "Não achei: $i"
    }
}

Get-ChildItem $tmp -Recurse -Directory -Filter "__pycache__" -Force -ErrorAction SilentlyContinue |
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
Get-ChildItem $tmp -Recurse -File -Force -ErrorAction SilentlyContinue |
    Where-Object { $_.Extension -eq ".pyc" -or $_.Name -eq ".DS_Store" } |
    Remove-Item -Force -ErrorAction SilentlyContinue

Compress-Archive -Path "$tmp\*" -DestinationPath $out
Remove-Item $tmp -Recurse -Force

$mb = [math]::Round((Get-Item $out).Length / 1MB, 2)
Write-Host "`n$out gerado: $mb MB"
if ($mb -gt 1) {
    Write-Warning "Passou de 1 MB. Confira se alguma pasta pesada entrou (ex.: dentro de scripts/ ou tests/)."
} else {
    Write-Host "OK: abaixo de 1 MB."
}