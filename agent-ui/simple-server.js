const express = require('express')
const path = require('path')
const app = express()
const port = 3002

// Serve static files from the .next directory
app.use('/_next', express.static(path.join(__dirname, '.next')))

// Serve static files from the public directory
app.use(express.static(path.join(__dirname, 'public')))

// Serve the main page
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '.next/server/app/index.html'))
})

app.listen(port, () => {
  console.log(`Server running at http://localhost:${port}`)
})
