"use client"

import { useEffect } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import { useAdGenerator } from "@/hooks/useAdGenerator"

// This is the main wizard component for ad image generation
export default function AdGeneratorWizard() {
  // Use our custom hook for all state management and API calls
  const {
    currentStep,
    isLoading,
    error,
    formData,
    updateFormData,
    clearError,
    nextStep,
    prevStep,
    selectModel,
    uploadContent,
    generatePrompt,
    refinePrompt,
    refineAndRegenerate,
    generateImage,
    resetSession,
    testConnection,
  } = useAdGenerator()

  // Test backend connection on component mount
  useEffect(() => {
    testConnection()
  }, [testConnection])

  // File upload handler
  const handleFileUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      if (file.type.startsWith('image/')) {
        updateFormData('uploadedImage', file)
      } else {
        alert('Please upload an image file (PNG, JPG, etc.)')
      }
    }
  }

  // Drag and drop handlers
  const handleDragOver = (event) => {
    event.preventDefault()
  }

  const handleDrop = (event) => {
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      updateFormData('uploadedImage', file)
    } else {
      alert('Please upload an image file (PNG, JPG, etc.)')
    }
  }

  // Handle model selection with API call
  const handleModelSelect = async (model) => {
    await selectModel(model)
  }

  // Handle moodboard file upload
  const handleMoodboardUpload = (event) => {
    const files = Array.from(event.target.files)
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    updateFormData('moodboardFiles', [...(formData.moodboardFiles || []), ...imageFiles])
  }

  // Handle moodboard file drop
  const handleMoodboardDrop = (event) => {
    event.preventDefault()
    const files = Array.from(event.dataTransfer.files)
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    updateFormData('moodboardFiles', [...(formData.moodboardFiles || []), ...imageFiles])
  }

  // Remove moodboard file
  const removeMoodboardFile = (index) => {
    const newFiles = formData.moodboardFiles.filter((_, i) => i !== index)
    updateFormData('moodboardFiles', newFiles)
  }

  // Handle reference file upload
  const handleReferenceUpload = (event) => {
    const files = Array.from(event.target.files)
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    updateFormData('referenceFiles', [...(formData.referenceFiles || []), ...imageFiles])
  }

  // Handle reference file drop
  const handleReferenceDrop = (event) => {
    event.preventDefault()
    const files = Array.from(event.dataTransfer.files)
    const imageFiles = files.filter(file => file.type.startsWith('image/'))
    updateFormData('referenceFiles', [...(formData.referenceFiles || []), ...imageFiles])
  }

  // Remove reference file
  const removeReferenceFile = (index) => {
    const newFiles = formData.referenceFiles.filter((_, i) => i !== index)
    updateFormData('referenceFiles', newFiles)
  }

  // Handle prompt generation
  const handleGeneratePrompt = async () => {
    await generatePrompt()
  }

  // Handle prompt refinement
  const handleRefinePrompt = async () => {
    if (formData.promptRefinement) {
      await refinePrompt(formData.promptRefinement)
    }
  }

  // Handle image generation
  const handleImageGeneration = async (imageModel) => {
    await generateImage(imageModel)
  }

  return (
    <div className="min-h-screen bg-background p-8">
      {/* Main container with max width and centered */}
      <div className="mx-auto max-w-4xl">
        
        {/* Error display */}
        {error && (
          <div className="mb-4 p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
            <p className="text-destructive text-sm">{error}</p>
            <Button 
              variant="outline" 
              size="sm" 
              onClick={clearError}
              className="mt-2"
            >
              Dismiss
            </Button>
          </div>
        )}

        {/* Loading overlay */}
        {isLoading && (
          <div className="fixed inset-0 bg-background/80 flex items-center justify-center z-50">
            <div className="bg-card p-6 rounded-lg shadow-lg">
              <p className="text-foreground">Processing...</p>
            </div>
          </div>
        )}
        
        {/* Progress indicator */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-foreground mb-4">
            Ad Image Generator
          </h1>
          <div className="flex items-center space-x-4">
            {[1, 2, 3, 4, 5].map((step) => (
              <div key={step} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    step <= currentStep
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground"
                  }`}
                >
                  {step}
                </div>
                {step < 5 && (
                  <div
                    className={`w-12 h-1 mx-2 ${
                      step < currentStep ? "bg-primary" : "bg-muted"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step 1: Session Setup */}
        {currentStep === 1 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Step 1: Choose AI Model</CardTitle>
              <CardDescription>
                Select the AI model for text analysis and prompt generation.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Model selection */}
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <input
                    type="radio"
                    id="openai"
                    name="model"
                    value="openai"
                    checked={formData.selectedModel === 'openai'}
                    onChange={() => handleModelSelect('openai')}
                    disabled={isLoading}
                    className="w-4 h-4 text-primary"
                  />
                  <Label htmlFor="openai" className="text-lg">
                    OpenAI (GPT-4.1)
                  </Label>
                </div>
                <div className="flex items-center space-x-3">
                  <input
                    type="radio"
                    id="google"
                    name="model"
                    value="google"
                    checked={formData.selectedModel === 'google'}
                    onChange={() => handleModelSelect('google')}
                    disabled={isLoading}
                    className="w-4 h-4 text-primary"
                  />
                  <Label htmlFor="google" className="text-lg">
                    Google (Gemini-2.5-flash)
                  </Label>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Step 2: Content Input */}
        {currentStep === 2 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Step 2: Upload Content</CardTitle>
              <CardDescription>
                Upload your product image and describe your vision.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-6">
                <div>
                  <Label htmlFor="product-image" className="text-sm font-medium">
                    Product Image (Required)
                  </Label>
                  <div 
                    className="mt-2 border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                    onDragOver={handleDragOver}
                    onDrop={handleDrop}
                    onClick={() => document.getElementById('file-input').click()}
                  >
                    {formData.uploadedImage ? (
                      <div className="space-y-2">
                        <p className="text-sm text-primary font-medium">✓ {formData.uploadedImage.name}</p>
                        <p className="text-xs text-muted-foreground">Click to change or drag another file</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-muted-foreground">Click to upload or drag and drop</p>
                        <p className="text-xs text-muted-foreground">PNG, JPG, GIF up to 10MB</p>
                      </div>
                    )}
                  </div>
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={handleFileUpload}
                    className="hidden"
                  />
                </div>
                
                <div>
                  <Label htmlFor="vision" className="text-sm font-medium">
                    Your Vision
                  </Label>
                  <Textarea
                    id="vision"
                    className="mt-2"
                    rows="4"
                    placeholder="Describe your vision for the ad..."
                    value={formData.visionText}
                    onChange={(e) => updateFormData('visionText', e.target.value)}
                  />
                </div>

                <div>
                  <Label className="text-sm font-medium">
                    Creativity Level: {formData.creativityLevel || 50}%
                  </Label>
                  <div className="mt-2 px-2">
                    <Slider
                      value={[formData.creativityLevel || 50]}
                      onValueChange={(value) => updateFormData('creativityLevel', value[0])}
                      min={0}
                      max={100}
                      step={10}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>Conservative</span>
                      <span>Balanced</span>
                      <span>Creative</span>
                    </div>
                  </div>
                </div>

                <div>
                  <Label className="text-sm font-medium">Focus Level: {formData.focusSlider}/10</Label>
                  <div className="mt-2 px-2">
                    <Slider
                      value={[formData.focusSlider]}
                      onValueChange={(value) => updateFormData('focusSlider', value[0])}
                      min={0}
                      max={10}
                      step={1}
                      className="w-full"
                    />
                    <div className="flex justify-between text-xs text-muted-foreground mt-1">
                      <span>Broad</span>
                      <span>Balanced</span>
                      <span>Focused</span>
                    </div>
                  </div>
                </div>

                {/* Moodboard Upload Section */}
                <div className="space-y-4">
                  <div>
                    <Label className="text-sm font-medium">Moodboard Images (Optional)</Label>
                    <p className="text-xs text-muted-foreground mt-1">
                      Upload reference images to inspire the ad style and mood
                    </p>
                  </div>
                  
                  <div
                    className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-primary/50 transition-colors cursor-pointer"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleMoodboardDrop}
                    onClick={() => document.getElementById('moodboard-input').click()}
                  >
                    {formData.moodboardFiles && formData.moodboardFiles.length > 0 ? (
                      <div className="space-y-2">
                        <p className="text-sm text-primary font-medium">
                          {formData.moodboardFiles.length} moodboard image(s) uploaded
                        </p>
                        <div className="flex flex-wrap gap-2 justify-center">
                          {formData.moodboardFiles.map((file, index) => (
                            <div key={index} className="flex items-center gap-1 bg-primary/10 px-2 py-1 rounded text-xs">
                              <span>{file.name}</span>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  removeMoodboardFile(index)
                                }}
                                className="text-red-500 hover:text-red-700"
                              >
                                ×
                              </button>
                            </div>
                          ))}
                        </div>
                        <p className="text-xs text-muted-foreground">Click to add more or drag and drop</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-muted-foreground">Click to upload or drag and drop moodboard images</p>
                        <p className="text-xs text-muted-foreground">PNG, JPG, GIF up to 10MB each</p>
                      </div>
                    )}
                    <input
                      id="moodboard-input"
                      type="file"
                      multiple
                      accept="image/*"
                      onChange={handleMoodboardUpload}
                      className="hidden"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={prevStep} disabled={isLoading}>
                Previous
              </Button>
              <Button onClick={handleGeneratePrompt} disabled={isLoading || !formData.uploadedImage || !formData.visionText}>
                Generate Prompt
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Step 3: Prompt Review */}
        {currentStep === 3 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Step 3: Review Prompt</CardTitle>
              <CardDescription>
                Review the generated prompt and choose to refine or approve.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Prompt review */}
              <div className="space-y-4">
                <div className="bg-muted p-4 rounded-lg">
                  {formData.generatedPrompt ? (
                    <p className="text-foreground whitespace-pre-wrap">
                      {formData.generatedPrompt}
                    </p>
                  ) : (
                    <p className="text-muted-foreground">
                      Click "Generate Prompt" to create your ad prompt...
                    </p>
                  )}
                </div>
                
                {/* Refinement input */}
                <div className="space-y-4">
                  <div>
                    <Label htmlFor="refinement" className="text-sm font-medium">
                      Refine Prompt (Optional)
                    </Label>
                    <Textarea
                      id="refinement"
                      className="mt-2"
                      rows="3"
                      placeholder="Describe how you'd like to refine the prompt..."
                      value={formData.promptRefinement}
                      onChange={(e) => updateFormData('promptRefinement', e.target.value)}
                    />
                  </div>
                  
                  <div>
                    <Label className="text-sm font-medium">
                      Focus Level: {formData.focusSlider}/10
                    </Label>
                    <div className="mt-2 px-2">
                      <Slider
                        value={[formData.focusSlider]}
                        onValueChange={(value) => updateFormData('focusSlider', value[0])}
                        min={0}
                        max={10}
                        step={1}
                        className="w-full"
                      />
                      <div className="flex justify-between text-xs text-muted-foreground mt-1">
                        <span>Broad</span>
                        <span>Balanced</span>
                        <span>Focused</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={prevStep} disabled={isLoading}>
                Previous
              </Button>
              <div className="space-x-3">
                {formData.promptRefinement && (
                  <Button 
                    variant="outline" 
                    onClick={() => handleRefinePrompt(formData.promptRefinement, formData.focusSlider)}
                    disabled={isLoading}
                  >
                    Refine
                  </Button>
                )}
                <Button 
                  onClick={nextStep}
                  disabled={isLoading || !formData.generatedPrompt}
                >
                  Approve & Continue
                </Button>
              </div>
            </CardFooter>
          </Card>
        )}

        {/* Step 4: Image Generation */}
        {currentStep === 4 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Step 4: Generate Image</CardTitle>
              <CardDescription>
                Choose the image generation model and create your ad.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Image generation model selection */}
              <div className="space-y-4">
                <div className="flex items-center space-x-3">
                  <input
                    type="radio"
                    id="openai-img"
                    name="image-model"
                    value="openai"
                    checked={formData.selectedImageModel === 'openai'}
                    onChange={(e) => updateFormData('selectedImageModel', e.target.value)}
                    className="w-4 h-4 text-primary"
                  />
                  <Label htmlFor="openai-img" className="text-lg">
                    OpenAI (GPT-Image-1)
                  </Label>
                </div>
                <div className="flex items-center space-x-3">
                  <input
                    type="radio"
                    id="google-img"
                    name="image-model"
                    value="google"
                    checked={formData.selectedImageModel === 'google'}
                    onChange={(e) => updateFormData('selectedImageModel', e.target.value)}
                    className="w-4 h-4 text-primary"
                  />
                  <Label htmlFor="google-img" className="text-lg">
                    Google (Gemini-2.5-flash-image)
                  </Label>
                </div>

                {/* Reference Images Upload Section */}
                <div className="space-y-4 pt-4 border-t">
                  <div>
                    <Label className="text-sm font-medium">Reference Images (Optional)</Label>
                    <p className="text-xs text-muted-foreground mt-1">
                      Upload reference images to guide the final image generation
                    </p>
                  </div>
                  
                  <div
                    className="border-2 border-dashed border-border rounded-lg p-6 text-center hover:border-primary/50 transition-colors cursor-pointer"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleReferenceDrop}
                    onClick={() => document.getElementById('reference-input').click()}
                  >
                    {formData.referenceFiles && formData.referenceFiles.length > 0 ? (
                      <div className="space-y-2">
                        <p className="text-sm text-primary font-medium">
                          {formData.referenceFiles.length} reference image(s) uploaded
                        </p>
                        <div className="flex flex-wrap gap-2 justify-center">
                          {formData.referenceFiles.map((file, index) => (
                            <div key={index} className="flex items-center gap-1 bg-primary/10 px-2 py-1 rounded text-xs">
                              <span>{file.name}</span>
                              <button
                                type="button"
                                onClick={(e) => {
                                  e.stopPropagation()
                                  removeReferenceFile(index)
                                }}
                                className="text-red-500 hover:text-red-700"
                              >
                                ×
                              </button>
                            </div>
                          ))}
                        </div>
                        <p className="text-xs text-muted-foreground">Click to add more or drag and drop</p>
                      </div>
                    ) : (
                      <div className="space-y-2">
                        <p className="text-muted-foreground">Click to upload or drag and drop reference images</p>
                        <p className="text-xs text-muted-foreground">PNG, JPG, GIF up to 10MB each</p>
                      </div>
                    )}
                    <input
                      id="reference-input"
                      type="file"
                      multiple
                      accept="image/*"
                      onChange={handleReferenceUpload}
                      className="hidden"
                    />
                  </div>
                </div>
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={prevStep} disabled={isLoading}>
                Previous
              </Button>
              <Button 
                onClick={() => handleImageGeneration(formData.selectedImageModel)}
                disabled={isLoading || !formData.selectedImageModel}
              >
                Generate Image
              </Button>
            </CardFooter>
          </Card>
        )}

        {/* Step 5: Result Display */}
        {currentStep === 5 && (
          <Card>
            <CardHeader>
              <CardTitle className="text-2xl">Step 5: Your Ad Image</CardTitle>
              <CardDescription>
                Here's your generated ad image!
              </CardDescription>
            </CardHeader>
            <CardContent>
              {/* Result display */}
              <div className="space-y-4">
                {formData.generatedImage ? (
                  <div className="space-y-4">
                    <img 
                      src={formData.generatedImage} 
                      alt="Generated Ad" 
                      className="w-full max-w-md mx-auto rounded-lg shadow-lg"
                    />
                    <p className="text-center text-sm text-muted-foreground">
                      Your AI-generated ad image is ready!
                    </p>
                  </div>
                ) : (
                  <div className="bg-muted p-8 rounded-lg text-center">
                    <p className="text-muted-foreground">Generated image will appear here...</p>
                  </div>
                )}
              </div>
            </CardContent>
            <CardFooter className="flex justify-between">
              <Button variant="outline" onClick={prevStep} disabled={isLoading}>
                Previous
              </Button>
              <Button onClick={resetSession} disabled={isLoading}>
                Start New Session
              </Button>
            </CardFooter>
          </Card>
        )}
      </div>
    </div>
  )
}
