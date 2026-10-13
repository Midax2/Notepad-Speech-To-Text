package com.pg.notepadstt.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.pg.notepadstt.ui.theme.ButtonBarBackground

@Composable
fun NoteItemBlock(
    fileName: String,
    onClick: () -> Unit,
    onDelete: () -> Unit
) {
    val screenWidth = LocalConfiguration.current.screenWidthDp.dp

    Row(
        modifier = Modifier
            .width(screenWidth * 0.7f)
            .padding(8.dp)
            .background(Color.LightGray, shape = RoundedCornerShape(12.dp))
            .clickable { onClick() }
            .padding(12.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = fileName,
            fontSize = 16.sp,
            color = Color.Black,
            modifier = Modifier.weight(1f)
        )

        IconButton(onClick = onDelete) {
            Icon(
                imageVector = Icons.Default.Close,
                contentDescription = "Delete",
                tint = Color.Red
            )
        }
    }
}
