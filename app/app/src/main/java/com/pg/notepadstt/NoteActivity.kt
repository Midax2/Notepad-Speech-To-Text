package com.pg.notepadstt

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp
import com.pg.notepadstt.screens.ButtonBar
import com.pg.notepadstt.ui.theme.NotepadSTTTheme

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
    NotepadSTTTheme {
        Column(
            modifier = Modifier
                .padding(16.dp),
            verticalArrangement = Arrangement.SpaceBetween
        ) {
            Text("Title")
            Spacer(modifier = Modifier.weight(1f))
            ButtonBar()
        }
    }
}