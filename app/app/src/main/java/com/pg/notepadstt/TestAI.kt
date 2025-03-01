package com.pg.notepadstt

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SpeechToTextProcessor(private val context: Context) {
    private lateinit var interpreter: Interpreter

    // Load TensorFlow Lite model from assets
    fun loadModel(modelFileName: String) {
        try {
            val modelBuffer = loadModelFile(modelFileName)
            val options = Interpreter.Options()
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e("SpeechToText", "Error loading model", e)
        }
    }

    // Load and process WAV file from assets
    private fun loadAudioFromAssets(fileName: String): FloatArray {
        val inputStream: InputStream = context.assets.open(fileName)
        val byteArray = inputStream.readBytes()

        // Skip WAV header (first 44 bytes) and convert PCM data to FloatArray
        val audioData = byteArray.sliceArray(44 until byteArray.size)
        val floatArray = FloatArray(audioData.size / 2) // 16-bit PCM, 2 bytes per sample

        for (i in floatArray.indices) {
            val low = audioData[i * 2].toInt() and 0xFF
            val high = audioData[i * 2 + 1].toInt() shl 8
            val sample = high or low
            floatArray[i] = sample / 32768.0f // Normalize to [-1, 1] like Librosa
        }

        return floatArray
    }

    // Run inference
    fun runInference(audioFileName: String) {
        val signal = loadAudioFromAssets(audioFileName)

        val inputArray = arrayOf<Any>(signal)
        val outputBuffer = IntBuffer.allocate(2000)

        val outputMap = mutableMapOf<Int, Any>()
        outputMap[0] = outputBuffer

        // Resize input tensor if needed
        interpreter.resizeInput(0, intArrayOf(signal.size))
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap)
        val outputSize = interpreter.getOutputTensor(0).shape()[0]
        val outputArray = IntArray(outputSize)
        outputBuffer.rewind()
        outputBuffer.get(outputArray)

        val finalResult = StringBuilder()
        for (i in 0 until outputSize) {
            if (outputArray[i] != 0) {
                finalResult.append(outputArray[i].toChar())
            }
        }

        Log.d("SpeechToText", "Decoded Output: $finalResult")
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
}
