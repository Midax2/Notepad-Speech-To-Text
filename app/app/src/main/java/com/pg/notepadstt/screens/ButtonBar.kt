package com.pg.notepadstt.screens

import android.util.Log
import android.widget.Button
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
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.pg.notepadstt.AudioRecorder
import com.pg.notepadstt.R
import com.pg.notepadstt.SpeechToTextProcessor
import com.pg.notepadstt.ui.theme.ButtonBarBackground


@Composable
fun ButtonBar(
    sttProcessor: SpeechToTextProcessor,
    recorder: AudioRecorder
){
    val textFromSpeech= remember { mutableStateOf("") }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(ButtonBarBackground),
        horizontalArrangement = Arrangement.SpaceEvenly
    ) {
        BottomBarButton(
            iconVector = Icons.Default.Check,
            onClickEvent = {  },
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
            onClickEvent = {  },
            name = "Clear"
        )

    }
}
