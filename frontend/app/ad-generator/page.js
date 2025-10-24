"use client"

import { useState } from "react"

// This is the main wizard component for ad image generation
export default function AdGeneratorWizard() {
  // React state management - this controls which step is visible
  // useState(1) means we start at step 1 (Session Setup)
  const [currentStep, setCurrentStep] = useState(1)

  // Function to move to the next step
  const nextStep = () => {
    setCurrentStep(currentStep + 1)
  }

  // Function to move to the previous step
  const prevStep = () => {
    setCurrentStep(currentStep - 1)
  }

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      {/* Main container with max width and centered */}
      <div className="mx-auto max-w-4xl">
        
        {/* Progress indicator */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-4">
            Ad Image Generator
          </h1>
          <div className="flex items-center space-x-4">
            {[1, 2, 3, 4, 5].map((step) => (
              <div key={step} className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                    step <= currentStep
                      ? "bg-blue-600 text-white"
                      : "bg-gray-300 text-gray-600"
                  }`}
                >
                  {step}
                </div>
                {step < 5 && (
                  <div
                    className={`w-12 h-1 mx-2 ${
                      step < currentStep ? "bg-blue-600" : "bg-gray-300"
                    }`}
                  />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step 1: Session Setup */}
        {currentStep === 1 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Step 1: Choose AI Model</h2>
            <p className="text-gray-600 mb-6">
              Select the AI model for text analysis and prompt generation.
            </p>
            
            {/* Model selection will go here */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <input
                  type="radio"
                  id="openai"
                  name="model"
                  value="openai"
                  className="w-4 h-4 text-blue-600"
                />
                <label htmlFor="openai" className="text-lg">
                  OpenAI (GPT-4.1)
                </label>
              </div>
              <div className="flex items-center space-x-3">
                <input
                  type="radio"
                  id="google"
                  name="model"
                  value="google"
                  className="w-4 h-4 text-blue-600"
                />
                <label htmlFor="google" className="text-lg">
                  Google (Gemini-2.5-flash)
                </label>
              </div>
            </div>

            {/* Navigation buttons */}
            <div className="flex justify-end mt-6">
              <button
                onClick={nextStep}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {/* Step 2: Content Input */}
        {currentStep === 2 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Step 2: Upload Content</h2>
            <p className="text-gray-600 mb-6">
              Upload your product image and describe your vision.
            </p>
            
            {/* Content input will go here */}
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Product Image (Required)
                </label>
                <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                  <p className="text-gray-500">Click to upload or drag and drop</p>
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Your Vision
                </label>
                <textarea
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  rows="4"
                  placeholder="Describe your vision for the ad..."
                />
              </div>
            </div>

            {/* Navigation buttons */}
            <div className="flex justify-between mt-6">
              <button
                onClick={prevStep}
                className="px-6 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
              >
                Previous
              </button>
              <button
                onClick={nextStep}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}

        {/* Step 3: Prompt Review */}
        {currentStep === 3 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Step 3: Review Prompt</h2>
            <p className="text-gray-600 mb-6">
              Review the generated prompt and choose to refine or approve.
            </p>
            
            {/* Prompt review will go here */}
            <div className="bg-gray-50 p-4 rounded-lg mb-6">
              <p className="text-gray-700">
                Generated prompt will appear here...
              </p>
            </div>

            {/* Navigation buttons */}
            <div className="flex justify-between mt-6">
              <button
                onClick={prevStep}
                className="px-6 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
              >
                Previous
              </button>
              <div className="space-x-3">
                <button className="px-6 py-2 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 transition-colors">
                  Refine
                </button>
                <button
                  onClick={nextStep}
                  className="px-6 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
                >
                  Approve & Generate
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 4: Image Generation */}
        {currentStep === 4 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Step 4: Generate Image</h2>
            <p className="text-gray-600 mb-6">
              Choose the image generation model and create your ad.
            </p>
            
            {/* Image generation will go here */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <input
                  type="radio"
                  id="openai-img"
                  name="image-model"
                  value="openai"
                  className="w-4 h-4 text-blue-600"
                />
                <label htmlFor="openai-img" className="text-lg">
                  OpenAI (GPT-Image-1)
                </label>
              </div>
              <div className="flex items-center space-x-3">
                <input
                  type="radio"
                  id="google-img"
                  name="image-model"
                  value="google"
                  className="w-4 h-4 text-blue-600"
                />
                <label htmlFor="google-img" className="text-lg">
                  Google (Gemini-2.5-flash-image)
                </label>
              </div>
            </div>

            {/* Navigation buttons */}
            <div className="flex justify-between mt-6">
              <button
                onClick={prevStep}
                className="px-6 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
              >
                Previous
              </button>
              <button
                onClick={nextStep}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
              >
                Generate Image
              </button>
            </div>
          </div>
        )}

        {/* Step 5: Result Display */}
        {currentStep === 5 && (
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4">Step 5: Your Ad Image</h2>
            <p className="text-gray-600 mb-6">
              Here's your generated ad image!
            </p>
            
            {/* Result display will go here */}
            <div className="bg-gray-50 p-8 rounded-lg text-center">
              <p className="text-gray-500">Generated image will appear here...</p>
            </div>

            {/* Navigation buttons */}
            <div className="flex justify-between mt-6">
              <button
                onClick={prevStep}
                className="px-6 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400 transition-colors"
              >
                Previous
              </button>
              <button className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                Start New Session
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
