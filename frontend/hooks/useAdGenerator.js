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

import { useState, useCallback, useEffect } from 'react'
import * as api from '@/lib/api'
import { API_BASE_URL } from '@/lib/api'

// Helper function to fix image URL if it starts with /static/ or /uploads/
function fixImageUrl(imageUrl) {
  if (!imageUrl) return imageUrl
  if (imageUrl.startsWith('/static/') || imageUrl.startsWith('/uploads/')) {
    return `${API_BASE_URL}${imageUrl}`
  }
  return imageUrl
}

// localStorage key for session ID
const SESSION_STORAGE_KEY = 'adSessionId'

export function useAdGenerator() {
  // Core state
  const [currentStep, setCurrentStep] = useState(1)
  const [userSessionId, setUserSessionId] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // History state
  const [historyEvents, setHistoryEvents] = useState([]) // Array of history event objects
  const [historyPage, setHistoryPage] = useState(1) // Current page number
  const [historyLimit, setHistoryLimit] = useState(20) // Events per page
  const [historyTotal, setHistoryTotal] = useState(0) // Total events count
  const [historyHasMore, setHistoryHasMore] = useState(false) // Pagination flag
  const [isLoadingHistory, setIsLoadingHistory] = useState(false) // Loading state for history

  // Form data state
  const [formData, setFormData] = useState({
    // Step 1: AI Model Selection
    selectedModel: '', // 'openai' or 'gemini'
    
    // Step 2: Content Input
    uploadedImage: null, // File object for product image
    visionText: '', // User's vision description
    moodboardFiles: [], // Array of moodboard image files
    focusSlider: 5, // Focus slider (0-10): 0=product focus, 10=scene focus
    
    // Step 4: Image Generation
    referenceFiles: [], // Array of reference image files
    
    // Step 3: Prompt Review
    generatedPrompt: '', // AI-generated prompt
    promptRefinement: '', // User's refinement request
    
    // Step 4: Image Generation
    selectedImageModel: '', // 'openai' or 'gemini' for image generation
    
    // Step 5: Results
    generatedImage: null, // Generated image URL or data
    productImagePreview: null, // Product image preview URL (for display)
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

  // History management functions - defined early to avoid dependency order issues
  // Helper function to refresh history (can be called from any function)
  const refreshHistory = useCallback(async (sessionId) => {
    if (!sessionId) return
    
    try {
      const historyResponse = await api.getSessionHistory(sessionId, 1, 20)
      setHistoryEvents(historyResponse.events || [])
      setHistoryPage(historyResponse.page || 1)
      setHistoryLimit(historyResponse.limit || 20)
      setHistoryTotal(historyResponse.total || 0)
      setHistoryHasMore(historyResponse.has_more || false)
    } catch (historyErr) {
      // If history fetch fails, don't block the workflow
      console.error('Failed to refresh history:', historyErr)
    }
  }, [])

  // Navigation functions
  const nextStep = useCallback(() => {
    clearError()
    
    // Step 1 → Step 2: Must have model selected
    if (currentStep === 1) {
      if (!formData.selectedModel) {
        setError('Please select a model before proceeding.')
        return
      }
      setCurrentStep(2)
      return
    }
    
    // Step 2 → Step 3: Allow if prompt exists (already completed), otherwise validate
    if (currentStep === 2) {
      if (formData.generatedPrompt) {
        // Step was already completed, allow navigation
        setCurrentStep(3)
        return
      }
      // Step not completed, validate before allowing
      if (!formData.uploadedImage || !formData.visionText) {
        setError('Please upload a product image and enter your vision before proceeding.')
        return
      }
      setCurrentStep(3)
      return
    }
    
    // Step 3 → Step 4: Allow if prompt exists (step was completed)
    if (currentStep === 3) {
      if (!formData.generatedPrompt) {
        setError('Please generate a prompt before proceeding.')
        return
      }
      setCurrentStep(4)
      return
    }
    
    // Step 4 → Step 5: Allow if image exists (already completed), otherwise validate
    if (currentStep === 4) {
      if (formData.generatedImage) {
        // Step was already completed, allow navigation
        setCurrentStep(5)
        return
      }
      // Step not completed, validate before allowing
      if (!formData.selectedImageModel) {
        setError('Please select an image generation model before proceeding.')
        return
      }
      setCurrentStep(5)
      return
    }
    
    // Default: just advance (shouldn't reach here for Step 5)
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
      
      // Store session ID in localStorage for page refresh support
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId)
      
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
  }, [clearError])

  // Step 2: Generate and review prompt
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
      
      // Refresh history to show the new "prompt_built" event
      await refreshHistory(userSessionId)
      
      // Advance to Step 3 after successful prompt generation
      setCurrentStep(3)
      
    } catch (err) {
      setError(`Failed to generate prompt: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, formData, updateFormData, clearError, refreshHistory])

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
      
      // Refresh history to show the new "prompt_refined" event
      await refreshHistory(userSessionId)
      
    } catch (err) {
      setError(`Failed to refine prompt: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, updateFormData, clearError, refreshHistory])

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
      
      // Fix image URL: if it starts with /static/, prepend API base URL
      const imageUrl = fixImageUrl(imageResult.image_url || imageResult.content)
      updateFormData('generatedImage', imageUrl)
      
      // Refresh history to show the new "prompt_refined" and "image_generated" events
      await refreshHistory(userSessionId)
      
    } catch (err) {
      setError(`Failed to refine and regenerate: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, formData.selectedImageModel, updateFormData, clearError, refreshHistory])

  // Step 4: Generate image
  const generateImage = useCallback(async (imageModel) => {
    try {
      setIsLoading(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session.')
      }

      updateFormData('selectedImageModel', imageModel)
      
      // Pass reference files to API
      const referenceFiles = formData.referenceFiles || []
      const imageResult = await api.generateImage(userSessionId, imageModel, referenceFiles)
      
      // Fix image URL: if it starts with /static/, prepend API base URL
      const imageUrl = fixImageUrl(imageResult.image_url || imageResult.content)
      updateFormData('generatedImage', imageUrl)
      
      // Set product image preview from backend (persistent source)
      // First try to get from session status (backend stores image_path)
      try {
        const status = await api.getSessionStatus(userSessionId)
        if (status.image_analysis?.image_path) {
          const productImageUrl = fixImageUrl(status.image_analysis.image_path)
          updateFormData('productImagePreview', productImageUrl)
        } else if (formData.uploadedImage) {
          // Fallback to client-side object URL if backend doesn't have it yet
          const productImageUrl = URL.createObjectURL(formData.uploadedImage)
          updateFormData('productImagePreview', productImageUrl)
        }
      } catch (statusErr) {
        // If session status fetch fails, fallback to client-side object URL
        if (formData.uploadedImage) {
          const productImageUrl = URL.createObjectURL(formData.uploadedImage)
          updateFormData('productImagePreview', productImageUrl)
        }
      }
      
      // Refresh history to show the new "image_generated" event
      await refreshHistory(userSessionId)
      
      // Auto-advance to next step
      nextStep()
    } catch (err) {
      setError(`Failed to generate image: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [userSessionId, formData.referenceFiles, formData.uploadedImage, updateFormData, nextStep, clearError, refreshHistory])

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

  // History management functions
  // Reset history state (called when starting new session)
  const resetHistory = useCallback(() => {
    setHistoryEvents([])
    setHistoryPage(1)
    setHistoryLimit(20)
    setHistoryTotal(0)
    setHistoryHasMore(false)
    setIsLoadingHistory(false)
  }, [])

  // Fetch history events for a specific page
  const fetchHistory = useCallback(async (page = 1, limit = 20) => {
    try {
      setIsLoadingHistory(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session. Cannot fetch history.')
      }

      // Call the API to get history
      const response = await api.getSessionHistory(userSessionId, page, limit)
      
      // Update state with response data
      // response contains: { events, total, page, limit, has_more }
      setHistoryEvents(response.events || [])
      setHistoryPage(response.page || page)
      setHistoryLimit(response.limit || limit)
      setHistoryTotal(response.total || 0)
      setHistoryHasMore(response.has_more || false)
      
    } catch (err) {
      setError(`Failed to fetch history: ${err.message}`)
    } finally {
      setIsLoadingHistory(false)
    }
  }, [userSessionId, clearError])

  // Load more history events (for pagination)
  const loadMoreHistory = useCallback(async () => {
    try {
      if (!historyHasMore || isLoadingHistory) {
        return // Don't load if no more pages or already loading
      }

      setIsLoadingHistory(true)
      clearError()
      
      if (!userSessionId) {
        throw new Error('No active session. Cannot load more history.')
      }

      // Calculate next page number
      const nextPage = historyPage + 1
      
      // Fetch next page
      const response = await api.getSessionHistory(userSessionId, nextPage, historyLimit)
      
      // Append new events to existing array (don't replace)
      setHistoryEvents(prev => [...prev, ...(response.events || [])])
      setHistoryPage(response.page || nextPage)
      setHistoryTotal(response.total || 0)
      setHistoryHasMore(response.has_more || false)
      
    } catch (err) {
      setError(`Failed to load more history: ${err.message}`)
    } finally {
      setIsLoadingHistory(false)
    }
  }, [userSessionId, historyPage, historyLimit, historyHasMore, isLoadingHistory, clearError])

  // Restore session state from backend (for page refresh support)
  const restoreSession = useCallback(async (sessionId) => {
    try {
      setIsLoading(true)
      clearError()
      
      // Fetch session status from backend
      const status = await api.getSessionStatus(sessionId)
      
      // Check if session exists (backend returns 200 with session_exists=false for missing sessions)
      if (!status.session_exists) {
        // Session doesn't exist (stale localStorage), clear it silently and start fresh
        localStorage.removeItem(SESSION_STORAGE_KEY)
        setCurrentStep(1)
        return
      }
      
      // Restore session ID and model provider
      setUserSessionId(status.session_id)
      updateFormData('selectedModel', status.model_provider)
      
      // Restore image analysis
      if (status.image_analysis) {
        updateFormData('imageAnalysis', status.image_analysis)
        // Convert image path to server URL for product image preview
        if (status.image_analysis.image_path) {
          const productImageUrl = fixImageUrl(status.image_analysis.image_path)
          updateFormData('productImagePreview', productImageUrl)
        }
      }
      
      // Restore user vision
      if (status.user_vision) {
        updateFormData('userVision', status.user_vision)
        // Restore original text input for display in Step 2
        if (status.user_vision.original_text) {
          updateFormData('visionText', status.user_vision.original_text)
        }
      }
      
      // Restore prompt
      if (status.prompt) {
        updateFormData('generatedPrompt', status.prompt.prompt_text)
        updateFormData('focusSlider', status.prompt.focus_slider)
      }
      
      // Restore generated image
      if (status.generated_image) {
        const imageUrl = fixImageUrl(status.generated_image.image_url)
        updateFormData('generatedImage', imageUrl)
        // Also restore selected image model if available
        if (status.generated_image.model_provider) {
          updateFormData('selectedImageModel', status.generated_image.model_provider)
        }
      }
      
      // Determine correct step based on what exists
      let targetStep = 1
      if (status.generated_image) {
        targetStep = 5 // Generated image exists -> Step 5
      } else if (status.prompt) {
        targetStep = 4 // Prompt exists but no image -> Step 4
      } else if (status.image_analysis || status.user_vision) {
        targetStep = 3 // Analysis exists but no prompt -> Step 3 (shouldn't happen, but handle it)
      } else if (status.session_id) {
        targetStep = 2 // Session exists but no analysis -> Step 2
      }
      
      setCurrentStep(targetStep)
      
    } catch (err) {
      // If restoration fails due to network/API error, clear localStorage and start fresh
      console.error('Failed to restore session:', err)
      localStorage.removeItem(SESSION_STORAGE_KEY)
      setError(`Failed to restore session: ${err.message}`)
    } finally {
      setIsLoading(false)
    }
  }, [clearError, updateFormData])

  // Restore session on mount if sessionId exists in localStorage
  useEffect(() => {
    const savedSessionId = localStorage.getItem(SESSION_STORAGE_KEY)
    if (savedSessionId && !userSessionId) {
      // Only restore if we don't already have a session (prevents double restoration)
      restoreSession(savedSessionId)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []) // Only run on mount - restoreSession is stable and doesn't need to be in deps

  // Reset everything for new session
  const resetSession = useCallback(() => {
    // Clean up object URL to prevent memory leaks
    setFormData(prev => {
      if (prev.productImagePreview && prev.productImagePreview.startsWith('blob:')) {
        URL.revokeObjectURL(prev.productImagePreview)
      }
      return {
        selectedModel: '',
        uploadedImage: null,
        visionText: '',
        moodboardFiles: [],
        focusSlider: 5,
        referenceFiles: [],
        generatedPrompt: '',
        promptRefinement: '',
        selectedImageModel: '',
        generatedImage: null,
        productImagePreview: null,
        imageAnalysis: null,
        userVision: null,
      }
    })
    setCurrentStep(1)
    setUserSessionId(null)
    setError(null)
    // Clear session ID from localStorage
    localStorage.removeItem(SESSION_STORAGE_KEY)
    // Reset history when starting new session
    resetHistory()
  }, [resetHistory])

  return {
    // State
    currentStep,
    userSessionId,
    isLoading,
    error,
    formData,
    
    // History state
    historyEvents,
    historyPage,
    historyLimit,
    historyTotal,
    historyHasMore,
    isLoadingHistory,
    
    // Actions
    updateFormData,
    clearError,
    nextStep,
    prevStep,
    selectModel,
    generatePrompt,
    refinePrompt,
    refineAndRegenerate,
    generateImage,
    resetSession,
    testConnection,
    
    // History actions
    fetchHistory,
    loadMoreHistory,
    resetHistory,
  }
}
