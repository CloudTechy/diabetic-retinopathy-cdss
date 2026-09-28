$authPath = Join-Path $env:APPDATA "com.vercel.cli\Data\auth.json"
$auth = Get-Content $authPath | ConvertFrom-Json
$token = $auth.token
Write-Host "Deploying to Vercel via global CLI..."
& "C:\Users\USER\AppData\Local\nvm\v24.9.0\vercel.cmd" deploy --prod --yes --token $token
