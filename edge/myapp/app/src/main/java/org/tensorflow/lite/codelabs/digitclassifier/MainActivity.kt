/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package org.tensorflow.lite.codelabs.digitclassifier

// CameraX imports
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import java.util.concurrent.Executors
import android.graphics.Bitmap
import android.graphics.BitmapFactory

import android.annotation.SuppressLint
import android.os.Bundle
import android.util.Log
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

  private lateinit var previewView: PreviewView
  private val cameraExecutor = Executors.newSingleThreadExecutor()
  private val REQUEST_CAMERA_PERMISSION = 1001
  private var predictedTextView: TextView? = null
  private var steeringModel = SteeringModel(this)

  @SuppressLint("ClickableViewAccessibility")
  override fun onCreate(savedInstanceState: Bundle?) {
    super.onCreate(savedInstanceState)
    setContentView(R.layout.activity_main)

    // Setup view instances.
    previewView = findViewById(R.id.previewView)
    predictedTextView = findViewById(R.id.predicted_text)

    // Setup steering model.
    steeringModel
      .initialize()
      .addOnFailureListener { e -> Log.e(TAG, "Error to setting up steering model.", e) }

    // Start camera.
    if (ContextCompat.checkSelfPermission(
        this,
        android.Manifest.permission.CAMERA
        ) == android.content.pm.PackageManager.PERMISSION_GRANTED) {
        startCamera()
    } else {
      requestPermissions(
        arrayOf(android.Manifest.permission.CAMERA),
        REQUEST_CAMERA_PERMISSION
      )
    }
  }

  override fun onRequestPermissionsResult(
    requestCode: Int,
    permissions: Array<out String>,
    grantResults: IntArray
  ) {
    super.onRequestPermissionsResult(requestCode, permissions, grantResults)

    if (requestCode == REQUEST_CAMERA_PERMISSION) {
      if (grantResults.isNotEmpty() && grantResults[0] == android.content.pm.PackageManager.PERMISSION_GRANTED) {
        startCamera()
      }
    }
  }

  override fun onDestroy() {
    // Sync DigitClassifier instance lifecycle with MainActivity lifecycle,
    // and free up resources (e.g. TF Lite instance) once the activity is destroyed.
    steeringModel.close()
    super.onDestroy()
  }

  private fun startCamera() {
    val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

    cameraProviderFuture.addListener({

      val cameraProvider = cameraProviderFuture.get()

      val preview = Preview.Builder().build().also {
        it.setSurfaceProvider(previewView.surfaceProvider)
      }

      val imageAnalyzer = ImageAnalysis.Builder()
        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
        .build()

      imageAnalyzer.setAnalyzer(cameraExecutor, FrameAnalyzer())

      val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

      cameraProvider.unbindAll()

      cameraProvider.bindToLifecycle(
        this,
        cameraSelector,
        preview,
        imageAnalyzer
      )

    }, ContextCompat.getMainExecutor(this))
  }

  inner class FrameAnalyzer : ImageAnalysis.Analyzer {
    override fun analyze(imageProxy: ImageProxy) {
      val bitmap = imageProxyToBitmap(imageProxy)

      if (bitmap != null && steeringModel.isInitialized) {
        steeringModel.classifyAsync(bitmap)
          .addOnSuccessListener { result ->

            runOnUiThread {
              predictedTextView?.text = result
            }
          }
      }
      imageProxy.close()
    }
  }

  private fun imageProxyToBitmap(image: ImageProxy): Bitmap? {
    val yBuffer = image.planes[0].buffer
    val uBuffer = image.planes[1].buffer
    val vBuffer = image.planes[2].buffer

    val ySize = yBuffer.remaining()
    val uSize = uBuffer.remaining()
    val vSize = vBuffer.remaining()

    val nv21 = ByteArray(ySize + uSize + vSize)

    yBuffer.get(nv21, 0, ySize)
    vBuffer.get(nv21, ySize, vSize)
    uBuffer.get(nv21, ySize + vSize, uSize)

    val yuvImage = android.graphics.YuvImage(
      nv21,
      android.graphics.ImageFormat.NV21,
      image.width,
      image.height,
      null
    )

    val out = java.io.ByteArrayOutputStream()
    yuvImage.compressToJpeg(
      android.graphics.Rect(0, 0, image.width, image.height),
      100,
      out
    )

    val imageBytes = out.toByteArray()

    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
  }

  companion object {
    private const val TAG = "MainActivity"
  }
}
