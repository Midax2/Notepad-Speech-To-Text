package com.pg.notepadstt

import android.content.Context
import android.util.Log
import com.jlibrosa.audio.JLibrosa
import org.tensorflow.lite.Interpreter
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.nio.IntBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * SpeechToTextProcessor is responsible for processing speech-to-text using a TensorFlow Lite model.
 * It loads an audio file, processes it, and runs inference to convert speech into text.
 */
class SpeechToTextProcessor(private val context: Context) {
    private lateinit var interpreter: Interpreter
    private lateinit var jLibrosa: JLibrosa

    // Constants for audio processing
    companion object {
        const val OUTPUT_BUFFER_CAPACITY = 50000
        const val SAMPLE_RATE = 16000
        const val READ_DURATION = -1

    }

    /**
     * Loads the TensorFlow Lite model from the assets folder.
     * @param modelFileName The name of the TFLite model file.
     */
    fun loadModel(modelFileName: String) {
        try {
            val modelBuffer = loadModelFile(modelFileName)
            val options = Interpreter.Options().setNumThreads(4).setUseXNNPACK(true)
            interpreter = Interpreter(modelBuffer, options)
        } catch (e: Exception) {
            Log.e("SpeechToText", "Error loading model", e)
        }
    }

    /**
     * Runs inference on the given audio file and converts speech into text.
     * @param audioFileName The name of the audio file stored in assets.
     * @since Audio file should be in WAV format and have 16kHz sample rate.
     * @return The converted text.
     */
    fun runInference(audioFileName: String) : String {
        try {
            jLibrosa = JLibrosa()
            val signal = jLibrosa
                .loadAndRead(copyWavFileToCache(audioFileName), SAMPLE_RATE, READ_DURATION)
            val inputArray = arrayOf<Any>(signal)
            val outputBuffer = IntBuffer.allocate(OUTPUT_BUFFER_CAPACITY)

            val outputMap = mutableMapOf<Int, Any>()
            outputMap[0] = outputBuffer

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
            return finalResult.toString()
        } catch (e: Exception) {
            throw Exception("Error running inference")
        }
    }

    /**
     * Copies a WAV file from assets to the cache directory.
     * @param wavFilename The name of the WAV file in assets.
     * @return The absolute path of the cached WAV file.
     */
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

    /**
     * Loads the TFLite model file from the assets folder.
     * @param modelName The name of the model file.
     * @return A MappedByteBuffer containing the model file.
     */
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = context.assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = fileDescriptor.startOffset
        val declaredLength = fileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }

    /**
     * Releases the TensorFlow Lite interpreter.
     */
    fun releaseInterpreter() {
        if (::interpreter.isInitialized) {
            interpreter.close()

        }
    }
}
