package com.pg.notepadstt

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import com.google.android.gms.tasks.Task
import com.google.android.gms.tflite.java.TfLite
import com.pg.notepadstt.ui.theme.NotepadSTTTheme
import org.tensorflow.lite.InterpreterApi
import java.io.File
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {

    private lateinit var interpreter: InterpreterApi
    private lateinit var modelBuffer: File
    private lateinit var executorService: ExecutorService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val fileDir = "model/1.tflite"
        executorService = Executors.newSingleThreadExecutor()
        modelBuffer = File(filesDir, fileDir)

        val initializeTask: Task<Void> = TfLite.initialize(this)
        initializeTask.addOnSuccessListener {
            executorService.execute {
                try {
                    val interpreterOption =
                        InterpreterApi.Options().setRuntime(InterpreterApi.Options.TfLiteRuntime.FROM_SYSTEM_ONLY)
                    interpreter = InterpreterApi.create(
                        modelBuffer,
                        interpreterOption
                    )
                    // Run the model after initialization
                    interpreter.run("input", "output")
                } catch (e: Exception) {
                    Log.e("Interpreter", "Error during model execution", e)
                }
            }
        }.addOnFailureListener { e ->
            Log.e("Interpreter", "Cannot initialize interpreter", e)
        }

        setContent {
            NotepadSTTTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding)
                    )
                }
            }
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        executorService.shutdown()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    Text(
        text = "Hello $name!",
        modifier = modifier
    )
}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    NotepadSTTTheme {
        Greeting("Android")
    }
}