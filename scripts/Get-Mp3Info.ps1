function Get-Mp3Info {
    param([string]$Path)

    $fs = [System.IO.File]::OpenRead($Path)
    $buf = New-Object byte[] 262144
    $n = $fs.Read($buf, 0, $buf.Length)
    $fs.Close()

    for ($i = 0; $i -lt $n - 4; $i++) {
        if ($buf[$i] -ne 0xFF) { continue }
        if (($buf[$i + 1] -band 0xE0) -ne 0xE0) { continue }

        $b1 = $buf[$i + 1]; $b2 = $buf[$i + 2]; $b3 = $buf[$i + 3]

        $verBits   = ($b1 -shr 3) -band 0x03
        $layerBits = ($b1 -shr 1) -band 0x03
        $brIndex   = ($b2 -shr 4) -band 0x0F
        $srIndex   = ($b2 -shr 2) -band 0x03
        $chanMode  = ($b3 -shr 6) -band 0x03

        if ($verBits -eq 1 -or $layerBits -eq 0 -or $srIndex -eq 3) { continue }

        $version = switch ($verBits) { 0 {"2.5"} 2 {"2"} 3 {"1"} }
        $srTable = @{
            "1"   = @(44100, 48000, 32000)
            "2"   = @(22050, 24000, 16000)
            "2.5" = @(11025, 12000, 8000)
        }
        $sampleRate = $srTable[$version][$srIndex]

        $brTable = @{
            "1"   = @(0,32,40,48,56,64,80,96,112,128,160,192,224,256,320)
            "2"   = @(0,8,16,24,32,40,48,56,64,80,96,112,128,144,160)
            "2.5" = @(0,8,16,24,32,40,48,56,64,80,96,112,128,144,160)
        }
        $bitrate = $brTable[$version][$brIndex] * 1000

        $channels = switch ($chanMode) { 3 {"1 (mono)"} 1 {"2 (joint stereo)"} default {"2"} }

        return [PSCustomObject]@{
            File       = Split-Path $Path -Leaf
            MPEG       = $version
            SampleRate = $sampleRate
            BitrateKbps = $bitrate / 1000
            Channels   = $channels
        }
    }
    return "no MPEG frame header found"
}

Get-Mp3Info "C:\Users\Barbara\labri_audio_files\stt\The Holy Spirit - part 1_ The Gifts of the Holy Spirit - Jerram Barrs___1931.mp3"
Get-Mp3Info "C:\Users\Barbara\labri_audio_files\stt\The Holy Spirit - part 1_ The Gifts of the Holy Spirit - Jerram Barrs___1931_0-5.mp3"
