package com.pg.notepadstt

import android.content.pm.PackageManager
import android.icu.text.CaseMap.Title
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.Button
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.pg.notepadstt.screens.ButtonBar
import com.pg.notepadstt.screens.EditableTextField
import com.pg.notepadstt.ui.theme.NotepadSTTTheme
import  android.Manifest
import android.widget.Button
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts

class NoteActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            NotepadSTTTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) {innerPading->
                    NotePreview(modifier = Modifier.padding(innerPading))
                }
            }
        }
    }
}



@Preview(showBackground = true)
@Composable
fun NotePreview(modifier: Modifier=Modifier) {
    val context= LocalContext.current
    val configuration= LocalConfiguration.current
    val screenHeight=configuration.screenHeightDp.dp
    val sttProcessor=SpeechToTextProcessor(context)
    val title = remember { mutableStateOf("") }
    val textState= remember { mutableStateOf("") }
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        )
    }
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { granted ->
        hasPermission = granted
    }
    //sttProcessor.loadModel("STT.tflite")
    NotepadSTTTheme {
        Column(
            modifier = Modifier
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Spacer(modifier=Modifier.height(screenHeight*0.15f))
            if(!hasPermission){
                Button(onClick= {
                    launcher.launch(Manifest.permission.RECORD_AUDIO)
                }){
                    Text("Grant Microphone Permission")
                }

            }else {
                val recorder = remember { AudioRecorder(context) }

                OutlinedTextField(
                    value = title.value,
                    onValueChange = { value -> title.value = value },
                    placeholder = { Text("Title") },
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(screenHeight * 0.05f), // Fixed height
                    singleLine = true
                )
                EditableTextField(textState)
                Spacer(modifier = Modifier.weight(1f))
                ButtonBar(sttProcessor, recorder, textState,title,context)
            }
        }
    }
}