package com.pg.notepadstt.screens

import android.content.Context
import android.util.Log
import android.widget.Button
import android.widget.Toast
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AddCircle
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Home
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.pg.notepadstt.AudioRecorder
import com.pg.notepadstt.R
import com.pg.notepadstt.SpeechToTextProcessor
import com.pg.notepadstt.ui.theme.ButtonBarBackground
import java.io.File
import java.io.IOException


@Composable
fun ButtonBar(
    sttProcessor: SpeechToTextProcessor,
    recorder: AudioRecorder,
    textFromSpeech: MutableState<String>,
    title:MutableState<String>,
    context: Context
){
    //val textFromSpeech= remember { mutableStateOf("") }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(ButtonBarBackground),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        BottomBarButton(
            iconVector = Icons.Default.Check,
            onClickEvent = {
                val fileName = title.value.trim().ifEmpty { "untitled" } + ".txt"
                val notesDir = File(context.filesDir, "notes")
                if (!notesDir.exists()) {
                    notesDir.mkdirs()
                }
                val file = File(notesDir, fileName)
                try {
                    file.writeText(textFromSpeech.value)
                    Log.i("FileSave", "Saved to: ${file.absolutePath}")
                    Toast.makeText(context, "Saved as ${file.name}", Toast.LENGTH_SHORT).show()
                } catch (e: IOException) {
                    e.printStackTrace()
                    Toast.makeText(context, "Failed to save file", Toast.LENGTH_SHORT).show()
                }
            },
            name = "Save"
        )
        BottomBarButton(
            iconInt = if (!recorder.isRecording.value) R.drawable.mic else null,
            iconVector = if (recorder.isRecording.value) Icons.Default.AddCircle else null,
            onClickEvent = {
                if (recorder.isRecording.value) {
                    recorder.stopRecording()
                    textFromSpeech.value=sttProcessor.runInference("temp_audio.wav")
                    sttProcessor.releaseInterpreter()
                    Log.i("Result Text from Speech:","Content:\" ${textFromSpeech.value}\"")
                }
                else{
                    sttProcessor.loadModel("STT.tflite")
                    recorder.startRecording()
                }
            },
            name = "Record"
        )

        BottomBarButton(
            iconVector = Icons.Default.Clear,
            onClickEvent = {
                textFromSpeech.value=""
            },
            name = "Clear"
        )

    }
}
