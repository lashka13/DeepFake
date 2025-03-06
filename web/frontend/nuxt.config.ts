export default defineNuxtConfig({
  compatibilityDate: '2024-11-01',
  devtools: { enabled: false },
  app: {
    head: {
      title: 'Big Juicy Models',
      meta: [
        { name: 'description', content: 'Big Juicy Models способен отличать настоящие изображения от сгенирированных, а также различать разных людей на фото' }
      ]
    }
  }
})
