/**
 * Custom React Hook for Ad Image Generation
 * 
 * This hook manages the entire ad generation workflow:
 * - Session management
 * - Form state
 * - API calls
 * - Loading states
 * - Error handling
 */

import { useState, useCallback } from 'react'
import * as api from '@/lib/api'

export function useAdGenerator() {
  // Core state
  const [currentStep, setCurrentStep] = useState(1)
  const [userSessionId, setUserSessionId] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Form data state
  const [formData, setFormData] = useState({
    // Step 1: AI Model Selection
    selectedModel: '', // 'openai' or 'google'
    
    // Step 2: Content Input
    uploadedImage: null, // File object for product image
    visionText: '', // User's vision description
    creativityLevel: 50, // Creativity slider (0-100)
    moodboardFiles: [], // Array of moodboard image files
    focusSlider: 5, // Focus slider (0-10)
    
    // Step 4: Image Generation
    referenceFiles: [], // Array of reference image files
    
    // Step 3: Prompt Review
    generatedPrompt: '', // AI-generated prompt
    promptRefinement: '', // User's refinement request
    
    // Step 4: Image Generation
    selectedImageModel: '', // 'openai' or 'google' for image generation
    
    // Step 5: Results
    generatedImage: null, // Generated image URL or data
    imageAnalysis: null, // Product image analysis
    userVision: null, // Parsed user vision
  })

  // Helper function to update form data
  const updateFormData = useCallback((field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }))
  }, [])

  // Clear error state
  const clearError = useCallback(() => {
    setError(null)
  }, [])

  // Navigation functions
  const nextStep = useCallback(() => {
    // Validation based on current step (Step 1 auto-advances)
    if (currentStep === 2 && !formData.uploadedImage) {
      setError('Please upload a product image before proceeding.')
      return
    }
    if (currentStep === 4 && !formData.selectedImageModel) {
      setError('Please select an image generation model before proceeding.')
      return
    }
    
    clearError()
    setCurrentStep(prev => prev + 1)
  }, [currentStep, formData, clearError])

  const prevStep = useCallback(() => {
    clearError()
    setCurrentStep(prev => prev - 1)
  }, [clearError])

  // Step 1: Create session and select model
  const selectModel = useCallback(async (model) => {
    try {
      setIsLoading(true)
      clearError()
      
      const sessionId = await api.createSession(model)
      setUserSessionId(sessionId)
      
      // Update form data and advance immediately
      setFormData(prev => ({
        ...prev,
        selectedModel: model
      }))
      
      // Auto-advance to next step
      setCurrentStep(2)
      
    } catch (err) {
      setError(`Failed to create session: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [])

  // Step 2: Upload content and analyze
  const uploadContent = useCallback(async (file, visionText) => {
    try {
      setIsLoading(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session. Please select a model first.')
      }

      // Update form data
      updateFormData('uploadedFile', file)
      updateFormData('visionText', visionText)

      // Analyze product image
      const imageAnalysis = await api.analyzeProductImage(file, userSessionId)
      updateFormData('imageAnalysis', imageAnalysis)

      // Parse user vision
      const userVision = await api.parseUserVision(visionText, userSessionId)
      updateFormData('userVision', userVision)

      // Auto-advance to next step
      nextStep()
    } catch (err) {
      setError(`Failed to process content: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, updateFormData, nextStep, clearError])

  // Step 3: Generate and review prompt
  const generatePrompt = useCallback(async () => {
    try {
      setIsLoading(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session.')
      }

      if (!formData.uploadedImage) {
        throw new Error('Please upload a product image first.')
      }

      if (!formData.visionText.trim()) {
        throw new Error('Please enter your vision text.')
      }

      const promptResult = await api.completePrompt(
        formData.uploadedImage,
        formData.visionText,
        formData.focusSlider,
        userSessionId,
        formData.moodboardFiles || []
      )
      updateFormData('generatedPrompt', promptResult.prompt_text || promptResult.content)
      
    } catch (err) {
      setError(`Failed to generate prompt: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, formData, updateFormData, clearError])

  // Step 3: Refine prompt
  const refinePrompt = useCallback(async (refinementText, focusSlider = null) => {
    try {
      setIsLoading(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session.')
      }

      const refinedPrompt = await api.refinePrompt(userSessionId, refinementText, focusSlider)
      updateFormData('generatedPrompt', refinedPrompt.prompt_text || refinedPrompt.content)
      updateFormData('promptRefinement', refinementText)
      
      if (focusSlider !== null) {
        updateFormData('focusSlider', focusSlider)
      }
      
    } catch (err) {
      setError(`Failed to refine prompt: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, updateFormData, clearError])

  // Refine prompt and regenerate image (from Step 5)
  const refineAndRegenerate = useCallback(async (refinementText) => {
    try {
      setIsLoading(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session.')
      }

      // First refine the prompt
      const refinedPrompt = await api.refinePrompt(userSessionId, refinementText)
      updateFormData('generatedPrompt', refinedPrompt.prompt_text || refinedPrompt.content)
      updateFormData('promptRefinement', refinementText)

      // Then regenerate the image with the same model
      const imageResult = await api.generateImage(userSessionId, formData.selectedImageModel)
      updateFormData('generatedImage', imageResult.image_url || imageResult.content)
      
    } catch (err) {
      setError(`Failed to refine and regenerate: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, formData.selectedImageModel, updateFormData, clearError])

  // Step 4: Generate image
  const generateImage = useCallback(async (imageModel) => {
    try {
      setIsLoading(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session.')
      }

      updateFormData('selectedImageModel', imageModel)
      
      const imageResult = await api.generateImage(userSessionId, imageModel)
      updateFormData('generatedImage', imageResult.image_url || imageResult.content)
      
      // Auto-advance to next step
      nextStep()
    } catch (err) {
      setError(`Failed to generate image: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, updateFormData, nextStep, clearError])

  // Reset everything for new session
  const resetSession = useCallback(() => {
    setCurrentStep(1)
    setUserSessionId(null)
    setError(null)
    setFormData({
      selectedModel: '',
      uploadedImage: null,
      visionText: '',
      creativityLevel: 50,
      moodboardFiles: [],
      focusSlider: 5,
      referenceFiles: [],
      generatedPrompt: '',
      promptRefinement: '',
      selectedImageModel: '',
      generatedImage: null,
      imageAnalysis: null,
      userVision: null,
    })
  }, [])

  // Test backend connection
  const testConnection = useCallback(async () => {
    try {
      const isConnected = await api.testConnection()
      if (!isConnected) {
        setError('Backend server is not responding. Please check if the server is running.')
      }
      return isConnected
    } catch (err) {
      setError('Failed to connect to backend server.')
      return false
    }
  }, [])

  return {
    // State
    currentStep,
    userSessionId,
    isLoading,
    error,
    formData,
    
    // Actions
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
  }
}
