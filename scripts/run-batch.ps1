param(
    [string]$Model  = 'models\ggml-small.en-q5_1.bin',
    [string]$Dir    = 'C:\Users\Barbara\labri_audio_files\stt\batch_run',
    [switch]$Force,
    [switch]$NoContext
)

$sttArgs = @('-m', $Model, '-dir', $Dir, '-all', '-audit', '0.2',
             '-corrections', 'dist\corrections.txt', '-on-interrupt', 'finish')
if ($Force) { $sttArgs += '-force' }
if ($NoContext) { $sttArgs += '-no-context' }

& "dist\transcriber.exe" @sttArgs
