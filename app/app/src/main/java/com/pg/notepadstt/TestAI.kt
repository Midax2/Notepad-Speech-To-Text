package com.pg.notepadstt

import android.content.Context
import android.util.Log
import com.jlibrosa.audio.JLibrosa
import com.jlibrosa.audio.wavFile.WavFile
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.IntBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class SpeechToTextProcessor(private val context: Context) {
    private lateinit var interpreter: Interpreter
    private lateinit var jLibrosa: JLibrosa

    // Load TensorFlow Lite model from assets
    fun loadModel(modelFileName: String) {
        try {
            val modelBuffer = loadModelFile(modelFileName)
            val options = Interpreter.Options().setNumThreads(4).setUseXNNPACK(true)
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e("SpeechToText", "Error loading model", e)
        }
    }

    // Run inference
    fun runInference(audioFileName: String) {
        jLibrosa = JLibrosa()
        val signal = jLibrosa.loadAndRead(copyWavFileToCache(audioFileName), 16000, -1)
        val inputArray = arrayOf<Any>(signal)
        val outputBuffer = IntBuffer.allocate(50000)

        val outputMap = mutableMapOf<Int, Any>()
        outputMap[0] = outputBuffer

        try {
            interpreter.resizeInput(0, intArrayOf(signal.size))
            interpreter.allocateTensors()
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
            interpreter.close()
        } catch (e: Exception) {
            Log.e("SpeechToText", "Error running inference", e)
            return
        }
    }

    private fun copyWavFileToCache(wavFilename: String): String {
        val destinationFile = File(context.cacheDir, wavFilename)
        if (!destinationFile.exists()) {
            try {
                context.assets.open(wavFilename).use { inputStream ->
                    val buffer = ByteArray(inputStream.available())
                    inputStream.read(buffer)

                    FileOutputStream(destinationFile).use { fileOutputStream ->
                        fileOutputStream.write(buffer)
                    }
                }
            } catch (e: Exception) {
                Log.e("SpeechToText", e.message ?: "Error copying WAV file to cache")
            }
        }
        return destinationFile.absolutePath
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
