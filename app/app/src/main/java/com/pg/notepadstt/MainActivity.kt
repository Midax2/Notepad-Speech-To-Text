package com.pg.notepadstt

import android.content.Intent
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.pg.notepadstt.screens.NoteItemBlock
import com.pg.notepadstt.ui.theme.NotepadSTTTheme
import java.io.File

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
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
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier) {
    val context= LocalContext.current
    val configuration= LocalConfiguration.current
    val screenHeight=configuration.screenHeightDp.dp
    val notesDir = File(context.filesDir, "notes")
    val files = remember {
        mutableStateListOf<File>().apply {
            addAll(notesDir.listFiles()?.toList() ?: emptyList())
        }
    }
    val scrollState = rememberScrollState()

    Column(
        modifier = Modifier
            .verticalScroll(scrollState)
            .padding(16.dp),

        verticalArrangement = Arrangement.SpaceBetween
    ) {
        Spacer(modifier = Modifier.height(screenHeight*0.05f))
        IconButton(
            onClick ={
                val intent = Intent(context, NoteActivity::class.java).apply {
                    putExtra("title", "")
                    putExtra("content", "")
                }
                context.startActivity(intent)
            }
        ) {
            Icon(
                imageVector = Icons.Default.Add,
                contentDescription = "NewNote"
            )
        }
        Spacer(modifier = Modifier.height(screenHeight*0.05f))
        if(files.size==0){
            Text("You Don't Have Notes Yet")
        }
        files.forEach { file ->

            Log.d("NotesFile", "File name: ${file.name}")
            NoteItemBlock(
                fileName = file.nameWithoutExtension,
                onClick = {
                    val intent = Intent(context, NoteActivity::class.java).apply {
                        putExtra("title", file.nameWithoutExtension)
                        putExtra("content", file.readText())
                    }
                    context.startActivity(intent)
                },
                onDelete = {
                    file.delete()
                    files.remove(file)

                }
            )

        }
        Spacer(modifier=Modifier.height(screenHeight*0.03f))

    }

}

@Preview(showBackground = true)
@Composable
fun GreetingPreview() {
    NotepadSTTTheme {
        Greeting("Android")
    }
}
