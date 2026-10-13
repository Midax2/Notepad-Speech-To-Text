package com.pg.notepadstt.screens

import android.util.Log
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonColors
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.platform.LocalConfiguration
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import com.pg.notepadstt.R
import com.pg.notepadstt.components.svgImageFromAssets
import com.pg.notepadstt.ui.theme.ButtonBarBackground
import com.pg.notepadstt.ui.theme.ButtonContentColor

@Composable
fun BottomBarButton(
    iconVector: ImageVector? = null,
    //iconName: String? = null,
    iconInt: Int?=null,
    onClickEvent: () -> Unit,
    name: String
) {
    val configuration= LocalConfiguration.current
    val screenWidth=configuration.screenWidthDp.dp
    Log.d("BottomBarButton:","Width=${screenWidth}")
    Button(
        onClick = onClickEvent,
        colors = ButtonDefaults.buttonColors(ButtonBarBackground)
        ) {
        if (iconVector != null) {
            Icon(imageVector = iconVector,
                contentDescription = name,
                tint = ButtonContentColor)
        } //else if (iconName != null) {
            else if(iconInt!=null){
            Icon(painter = painterResource(id=iconInt)/*svgImageFromAssets(iconName.toString())*/,
                contentDescription = name,
                modifier = Modifier.size(screenWidth*0.05f),
                tint = ButtonContentColor)
        }

        if(screenWidth>=410.dp) {
            Spacer(modifier = Modifier.width(screenWidth * 0.01f))

            Text(name, color = ButtonContentColor)
        }

    }
}
