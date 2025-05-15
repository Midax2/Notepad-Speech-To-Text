package com.pg.notepadstt.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.material3.TextField
import androidx.compose.runtime.Composable
import androidx.compose.runtime.MutableState
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pg.notepadstt.ui.theme.TextFieldBackgroundColor

@Composable
fun EditableTextField(textState: MutableState<String>){
    val configuration= LocalConfiguration.current
    val screenHeight=configuration.screenHeightDp.dp
    //val textState = remember { mutableStateOf("") }
    TextField(
        value=textState.value,
        onValueChange={value -> textState.value=value},
        placeholder = {Text("Text gonna appear here")},
        textStyle = TextStyle(fontSize = 14.sp, color = Color.Black),
        modifier = Modifier
            .fillMaxWidth()
            .background(TextFieldBackgroundColor)
            .heightIn(min =screenHeight*0.5f, max=screenHeight*0.7f),
        shape = RoundedCornerShape(9.0f)


    )
}