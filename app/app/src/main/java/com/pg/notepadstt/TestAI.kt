package com.pg.notepadstt

import android.util.Log
import androidx.activity.ComponentActivity
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import org.tensorflow.lite.InterpreterApi
import java.io.FileInputStream
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class TestAI : ComponentActivity() {

    private lateinit var interpreter: InterpreterApi
    private lateinit var executorService: ExecutorService

    fun run() {
        executorService = Executors.newSingleThreadExecutor()

        val initializeTask: Task<Void> = TfLite.initialize(this)
        initializeTask.addOnSuccessListener {
            executorService.execute {
                try {
                    val modelBuffer = loadModelFile("STT.tflite")
                    val interpreterOption =
                        InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                    interpreter = InterpreterApi.create(modelBuffer, interpreterOption)
                    // Run the model after initialization
                    val input = FloatArray(1) { 1.0f } // Placeholder input
                    val output = FloatArray(1)
                    interpreter.run(input, output)
                    Log.d("Interpreter", "Model executed successfully: ${output[0]}")
                } catch (e: Exception) {
                    Log.e("Interpreter", "Error during model execution", e)
                }
            }
        }.addOnFailureListener { e ->
            Log.e("Interpreter", "Cannot initialize interpreter", e)
        }
    }

    private fun loadModelFile(modelFileName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelFileName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, fileDescriptor.startOffset, fileDescriptor.declaredLength)
    }

    override fun onDestroy() {
        super.onDestroy()
        executorService.shutdown()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }
}