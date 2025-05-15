package com.pg.notepadstt.screens

import android.widget.Button
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.Home
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
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
            iconInt = R.drawable.mic,
            onClickEvent = {  },
            name = "Record"
        )

        BottomBarButton(
            iconVector = Icons.Default.Clear,
            onClickEvent = {  },
            name = "Clear"
        )

    }
}
