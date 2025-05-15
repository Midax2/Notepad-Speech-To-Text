package com.pg.notepadstt

import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import androidx.core.content.ContextCompat
import android.Manifest
import android.util.Log
import androidx.compose.runtime.mutableStateOf
import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.concurrent.thread

class AudioRecorder(private val context:Context) {
    private val SAMPLE_RATE=16000
    private val CHANEL_CONFIG = AudioFormat.CHANNEL_IN_STEREO
    private val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
    private val BUFFER_SIZE = AudioRecord.getMinBufferSize(SAMPLE_RATE,
        CHANEL_CONFIG, AUDIO_FORMAT)

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    val isRecording = mutableStateOf(false)


    private val pcmFile: File by lazy {
        File(context.filesDir, "temp_audio.pcm")
    }

    val wavFile: File by lazy {
        File(context.filesDir, "temp_audio.wav")
    }

    fun startRecording():File?{
        if (ContextCompat.checkSelfPermission(context, Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED
        ) {
            // You can throw, return null, or handle it gracefully
            Log.e("AudioRecorder", "RECORD_AUDIO permission not granted")
            return null
        }

        try {
            if(wavFile.exists())
                wavFile.delete()
            audioRecord=AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE,
                CHANEL_CONFIG,
                AUDIO_FORMAT,
                BUFFER_SIZE
            )
            if (pcmFile.exists()) pcmFile.delete()
            if (wavFile.exists()) wavFile.delete()

            audioRecord?.startRecording()
            isRecording.value=true
            Log.i("AudioRecorder:","Start recording ")
            recordingThread = thread {
                writePcmDataToFile()

            }

            return wavFile
        }
        catch (e: SecurityException){
            Log.e("AudioRecorder","SecurityException: ${e.message}")
            return null
        }

    }

    fun stopRecording(): File {
        isRecording.value = false
        audioRecord?.apply {
            stop()
            release()
        }
        recordingThread?.join()

        convertPcmToWav(pcmFile, wavFile)
        pcmFile.delete() // Clean up
        Log.i("AudioRecorder:","stop recording ${wavFile.exists()} ${wavFile.absolutePath}")
        return wavFile
    }

    private fun writePcmDataToFile() {
        val buffer = ByteArray(BUFFER_SIZE)
        FileOutputStream(pcmFile).use { os ->
            while (isRecording.value) {
                val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
                if (read > 0) {
                    os.write(buffer, 0, read)
                }
            }
        }
    }

    private fun convertPcmToWav(pcmFile: File, wavFile: File) {
        val pcmData = pcmFile.readBytes()
        val wavData = ByteArrayOutputStream()

        val totalDataLen = pcmData.size + 36
        val byteRate = SAMPLE_RATE * 2

        DataOutputStream(wavData).apply {
            writeBytes("RIFF")
            writeIntLE(totalDataLen)
            writeBytes("WAVE")
            writeBytes("fmt ")
            writeIntLE(16)
            writeShortLE(1)
            writeShortLE(1)
            writeIntLE(SAMPLE_RATE)
            writeIntLE(byteRate)
            writeShortLE(2)
            writeShortLE(16)
            writeBytes("data")
            writeIntLE(pcmData.size)
            write(pcmData)
        }

        wavFile.writeBytes(wavData.toByteArray())
    }


    private fun DataOutputStream.writeIntLE(value: Int) {
        write(ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putInt(value).array())
    }

    private fun DataOutputStream.writeShortLE(value: Int) {
        write(ByteBuffer.allocate(2).order(ByteOrder.LITTLE_ENDIAN).putShort(value.toShort()).array())
    }


    fun getRecordState(): Boolean = isRecording.value
}