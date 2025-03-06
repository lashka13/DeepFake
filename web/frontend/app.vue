<template>
  <div class="container">
    <div class="header">
      <h1>Big Juicy Models</h1>
      <p class="description">Загрузите две картинки с лицами и мы скажем насколько они похожи!</p>
    </div>
    <div class="main">
      <div class="upload-container">
        <div class="upload-box">
          <img v-if="image1Preview" :src="image1Preview" class="preview" />
          <div v-else class="upload-placeholder">
            <i class="fas fa-upload"></i>
            <p>Загрузите первое фото</p>
          </div>
          <input type="file" @change="handleImage1Upload" accept="image/*" class="file-input" ref="image1Input" />
        </div>

        <div class="upload-box">
          <img v-if="image2Preview" :src="image2Preview" class="preview" />
          <div v-else class="upload-placeholder">
            <i class="fas fa-upload"></i>
            <p>Загрузите второе фото</p>
          </div>
          <input type="file" @change="handleImage2Upload" accept="image/*" class="file-input" ref="image2Input" />
        </div>
      </div>

      <button @click="compareImages" class="compare-button" :disabled="!image1Preview || !image2Preview">
        <i class="fas fa-equals"></i>
        Сравнить лица
      </button>

      <div v-if="result" class="result" :class="{ match: (isMatch > 70000) }">
        <div class="result-content">
          <i :class="['fas', (isMatch > 70000) ? 'fa-check-circle' : 'fa-times-circle']"></i>
          <p class="result-message">{{ resultMessage }}</p>
          <p class="result-confidence">Уровень схожести: {{ isMatch / 1000 }}%</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'

const image1Preview = ref<string>('')
const image2Preview = ref<string>('')
const result = ref(false)
const isMatch = ref(0)
const resultMessage = ref('')
const image1Input = ref<HTMLInputElement | null>(null)
const image2Input = ref<HTMLInputElement | null>(null)

const handleImage1Upload = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    image1Preview.value = URL.createObjectURL(file)
  }
}

const handleImage2Upload = (event: Event) => {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    image2Preview.value = URL.createObjectURL(file)
  }
}

const compareImages = async () => {
  result.value = true
  const formData = new FormData()

  // Get the file objects from the file input refs
  const file1 = image1Input.value?.files?.[0]
  const file2 = image2Input.value?.files?.[0]

  // Append the image files to FormData
  formData.append('first_image', file1)
  formData.append('second_image', file2)

  // Send POST request to the backend API
  const response = await fetch('http://localhost:8000/compare_images/', {
    method: 'POST',
    body: formData,
  })

  if (response.ok) {
    const data = await response.json()
    isMatch.value = data.similarity // Temporary random result
    resultMessage.value = (isMatch.value > 70000) ? "Это один человек" : "Это разные люди"
  }
}
</script>

<style scoped>
.container {
  max-width: 1000px;
  margin: 0 auto;
  margin-top: 10px;
  padding: 2rem;
  font-family: 'Arial', sans-serif;
  border-radius: 25px;
  box-shadow: 0 4px 15px rgba(41, 128, 185, 0.3);
  background: linear-gradient(135deg, #ffffff, #f8f9fa);
}

.header {
  text-align: center;
  margin-bottom: 3rem;
}

h1 {
  font-size: 3.5rem;
  background: linear-gradient(135deg, #2980b9, #3498db);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 1rem;
}

.description {
  color: #666;
  font-size: 1.1rem;
}

.upload-container {
  display: flex;
  justify-content: center;
  gap: 2rem;
  margin-bottom: 2rem;
}

@media (max-width: 700px) {
  .upload-container {
    flex-direction: column;
    align-items: center;
  }

  .upload-box {
    width: 250px !important;
    height: 250px !important;
  }
}

.upload-box {
  width: 300px;
  height: 300px;
  border: 2px dashed #3498db;
  border-radius: 15px;
  position: relative;
  overflow: hidden;
  cursor: pointer;
  transition: all 0.3s ease;
  background: linear-gradient(135deg, #ffffff, #f8f9fa);
  box-shadow: 0 4px 15px rgba(41, 128, 185, 0.1);
}

.upload-box:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(41, 128, 185, 0.2);
  border-color: #2980b9;
}

.upload-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 100%;
  color: #3498db;
}

.upload-placeholder i {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.preview {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.file-input {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  opacity: 0;
  cursor: pointer;
}

.compare-button {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.75rem;
  margin: 0 auto;
  padding: 1rem 2rem;
  font-size: 1.2rem;
  font-weight: 600;
  background: linear-gradient(135deg, #2980b9, #3498db, #2980b9);
  background-size: 200% 100%;
  color: white;
  border: none;
  border-radius: 30px;
  cursor: pointer;
  transition: all 0.4s ease;
  box-shadow: 0 4px 15px rgba(41, 128, 185, 0.3);
  animation: gradientMove 3s ease infinite;
}

@keyframes gradientMove {
  0% {
    background-position: 0% 50%;
  }

  50% {
    background-position: 100% 50%;
  }

  100% {
    background-position: 0% 50%;
  }
}

.compare-button:hover:not(:disabled) {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(41, 128, 185, 0.4);
}

.compare-button:active:not(:disabled) {
  transform: translateY(1px);
  box-shadow: 0 2px 10px rgba(41, 128, 185, 0.2);
}

.compare-button:disabled {
  background: linear-gradient(135deg, #bdc3c7, #95a5a6);
  cursor: not-allowed;
  transform: none;
  box-shadow: none;
  opacity: 0.7;
}

.compare-button i {
  font-size: 1.2rem;
}

.result {
  margin-top: 2rem;
  text-align: center;
  padding: 2rem;
  border-radius: 15px;
  background: linear-gradient(135deg, #ff6b6b, #ff8787, #ffa8a8);
  background-size: 200% 100%;
  color: white;
  box-shadow: 0 4px 15px rgba(255, 107, 107, 0.3);
  transition: all 0.3s ease;
  animation: slideIn 0.5s ease-out, gradientMove 3s ease infinite;
}

@keyframes slideIn {
  from {
    transform: translateY(20px);
    opacity: 0;
  }

  to {
    transform: translateY(0);
    opacity: 1;
  }
}

.result.match {
  background: linear-gradient(135deg, #40c057, #51cf66, #69db7c);
  background-size: 200% 100%;
  box-shadow: 0 4px 15px rgba(64, 192, 87, 0.3);
}

.result-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.5rem;
}

.result-content i {
  font-size: 3rem;
  margin-bottom: 0.5rem;
}

.result-message {
  font-size: 1.5rem;
  font-weight: bold;
  margin: 0;
}

.result-confidence {
  font-size: 1rem;
  opacity: 0.9;
  margin: 0;
}
</style>
