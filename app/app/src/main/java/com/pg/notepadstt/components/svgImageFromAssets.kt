package com.pg.notepadstt.components

import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.painter.Painter
import androidx.compose.ui.platform.LocalContext
import coil.compose.rememberAsyncImagePainter
import coil.decode.SvgDecoder
import coil.request.ImageRequest

@Composable
fun svgImageFromAssets(fileName: String): Painter {
    return rememberAsyncImagePainter(
        ImageRequest.Builder(LocalContext.current)
            .data("file:///android_asset/$fileName") // Load from assets
            .decoderFactory(SvgDecoder.Factory()) // SVG Decoder
            .build()
    )
}
