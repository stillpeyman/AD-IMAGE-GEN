# Frontend History Integration - Task Prompt

## Context: What Has Been Completed

### Backend Implementation (✅ Complete)
- **History Endpoint**: `GET /sessions/{user_session_id}/history` implemented in `main.py` (lines 792-860)
- **Endpoint Features**:
  - Pagination support: `page` (default: 1) and `limit` (default: 20, max: 100) as query parameters
  - Returns paginated history events in chronological order (oldest first)
  - Response format:
    ```json
    {
      "events": ["formatted event string 1", "formatted event string 2", ...],
      "total": 100,
      "page": 1,
      "limit": 20,
      "has_more": true
    }
    ```
- **Event Formatting**: Helper function `format_event_text()` in `main.py` (lines 688-788) converts `HistoryEvent` records to chat-style readable text
- **Event Types Supported**: All workflow events (session_created, product_image_upload, product_image_analyzed, moodboard_upload, moodboard_image_analyzed, user_vision_submitted, vision_parsed, prompt_built, prompt_refined, prompt_refinement_request, image_model_chosen, reference_image_upload, image_generated)

### Frontend Structure (Current State)
- **Framework**: Next.js 15.5.6 with React 19.1.0
- **API Layer**: `frontend/lib/api.js` - centralized API calls with `apiCall()` wrapper
- **Custom Hook**: `frontend/hooks/useAdGenerator.js` - manages workflow state and API calls
- **Main Component**: `frontend/app/ad-generator/page.js` - wizard-style UI for ad generation
- **UI Components**: shadcn/ui components in `frontend/components/ui/` (Button, Card, Input, Label, Textarea, Slider)
- **Session Management**: `userSessionId` stored in `useAdGenerator` hook state

## Task: Integrate History Endpoint into Frontend

### Objective
Add a History UI component that displays session history events in a chat-style format with pagination controls. Users should be able to view their workflow history chronologically.

### Required Files

#### Backend (Reference - Already Complete)
- `main.py` (lines 688-860): History endpoint and formatting helper

#### Frontend (To Create/Modify)
1. **`frontend/lib/api.js`** (MODIFY)
   - Add `getSessionHistory(userSessionId, page, limit)` function
   - Follow existing API call pattern using `apiCall()` wrapper
   - Endpoint: `GET /sessions/${userSessionId}/history?page=${page}&limit=${limit}`

2. **`frontend/hooks/useAdGenerator.js`** (MODIFY)
   - Add history state management:
     - `historyEvents: []` - array of formatted event strings
     - `historyPage: number` - current page number
     - `historyLimit: number` - events per page
     - `historyTotal: number` - total events count
     - `historyHasMore: boolean` - pagination flag
     - `isLoadingHistory: boolean` - loading state
   - Add `fetchHistory(page, limit)` function to call API
   - Add `loadMoreHistory()` function for pagination
   - Add `resetHistory()` function to clear history state

3. **`frontend/components/History.jsx`** (CREATE NEW)
   - New component for displaying history events
   - Chat-style UI (messages in chronological order)
   - Display formatted event strings (already formatted by backend)
   - Pagination controls:
     - "Load More" button (when `has_more === true`)
     - "Previous Page" button (when `page > 1`)
     - Display "Page X of Y" information
   - Loading state indicator
   - Error state handling
   - Use existing UI components (Card, Button) from `components/ui/`

4. **`frontend/app/ad-generator/page.js`** (MODIFY)
   - Add History component to the wizard UI
   - Decide placement: separate step, sidebar, or modal
   - Add button/trigger to show history (e.g., "View History" button)
   - Pass `userSessionId` from hook to History component

### Implementation Requirements

#### API Integration (`frontend/lib/api.js`)
- Follow existing pattern: use `apiCall()` wrapper for GET requests
- Handle query parameters: `?page=${page}&limit=${limit}`
- Error handling: throw errors for frontend to catch
- Return full response object: `{ events, total, page, limit, has_more }`

#### State Management (`frontend/hooks/useAdGenerator.js`)
- Add history state variables (as listed above)
- `fetchHistory(page, limit)`: 
  - Set `isLoadingHistory = true`
  - Call `api.getSessionHistory(userSessionId, page, limit)`
  - Update state with response
  - Handle errors
  - Set `isLoadingHistory = false`
- `loadMoreHistory()`: 
  - Increment `historyPage`
  - Call `fetchHistory(newPage, historyLimit)`
  - Append new events to existing `historyEvents` array (don't replace)
- `resetHistory()`: Clear all history state when starting new session

#### History Component (`frontend/components/History.jsx`)
- **Display Format**: Chat-style messages (similar to chat interface)
  - Each event as a message bubble/card
  - Chronological order (oldest first, newest last)
  - Format: Use `\n` in event strings for line breaks (frontend will handle with CSS or conversion)
- **Pagination**:
  - "Load More" button: Only show when `has_more === true`
  - "Previous Page" button: Only show when `page > 1`
  - Display: "Page {page} of {Math.ceil(total/limit)}" or "Showing {events.length} of {total} events"
- **Loading State**: Show spinner/loading indicator when `isLoadingHistory === true`
- **Error State**: Display error message if history fetch fails
- **Styling**: Use Tailwind CSS (already configured), match existing UI style

#### Integration (`frontend/app/ad-generator/page.js`)
- Add History component to appropriate location in wizard
- Options:
  - **Option A**: Separate step in wizard (Step 6 or 7)
  - **Option B**: Sidebar/panel that can be toggled
  - **Option C**: Modal/dialog triggered by "View History" button
- Pass required props: `userSessionId`, history state, and functions from hook
- Only show history if `userSessionId` exists (session is active)

### Technical Specifications

#### API Endpoint Details
- **URL**: `GET /sessions/{user_session_id}/history?page={page}&limit={limit}`
- **Base URL**: `http://localhost:5001` (defined in `api.js`)
- **Query Parameters**:
  - `page`: integer, default 1
  - `limit`: integer, default 20, max 100
- **Response**: JSON object with `events`, `total`, `page`, `limit`, `has_more`
- **Event Format**: Array of strings (already formatted by backend)

#### Frontend Patterns to Follow
- **API Calls**: Use `apiCall()` wrapper from `api.js` (see existing functions)
- **State Management**: Use React hooks (`useState`, `useCallback`) in `useAdGenerator.js`
- **Component Structure**: Follow existing component patterns (see `page.js`)
- **Styling**: Use Tailwind CSS classes, match existing UI components
- **Error Handling**: Display user-friendly error messages
- **Loading States**: Show loading indicators during API calls

### User's Working Style Instructions

**CRITICAL: Follow these instructions exactly. The user has emphasized these requirements multiple times.**

1. **Step-by-Step Guidance**: 
   - Guide step-by-step, waiting for approval before proceeding
   - Do NOT show full solutions unless explicitly asked
   - Break tasks into small, manageable steps
   - Wait for user to complete each step before moving to next

2. **Do NOT Solve Tasks For User**:
   - Do NOT write complete code solutions
   - Do NOT make changes without explicit approval
   - Guide the user to write the code themselves
   - Explain concepts and patterns, let user implement

3. **Explain Before Implementing**:
   - Always explain reasoning and approach BEFORE suggesting code
   - Explain WHY something is done a certain way
   - Provide clear, beginner-friendly explanations
   - Use real-life analogies when helpful

4. **Code Changes**:
   - Do NOT change code/files unless explicitly asked
   - When suggesting changes, explain WHY using clear reasoning
   - Show before/after diffs when proposing changes
   - Wait for user approval with 'keep' or 'y' before applying

5. **Communication Style**:
   - Direct, token-efficient communication
   - NO emojis, affirmations, or praise
   - Get straight to the point
   - Focus on substance over style
   - NO saying "excellent" or similar positive reinforcement

6. **Documentation and Learning**:
   - Provide documentation links when explaining concepts
   - Refer to relevant documentation for learning
   - Explain each line of code and why it's done that way
   - Use actual code examples from the project, not isolated snippets

7. **Best Practices**:
   - Keep solutions simple, avoid overcomplicating
   - Focus on MVP-level implementation
   - Test with existing MVP foundation
   - Maintain code style consistency (PEP8 for Python, existing patterns for JavaScript)
   - Warn in advance if something is unnecessary or overcomplicating for MVP

8. **Clarity and Honesty**:
   - If unclear about user's point or idea, ASK for clarification
   - Do NOT make things up
   - Provide honest feedback and criticism
   - Explanations should be sharp and focused, avoid confusion

9. **Code Examples**:
   - Code examples should be actual puzzle pieces fitting into overall project workflow
   - Use user's existing code patterns and style
   - Reference existing code in the project
   - Show how new code integrates with existing code

10. **Testing Approach**:
    - User prefers to test working foundation before adding advanced features
    - Everything must work with existing MVP foundation
    - Serve future development steps

### Implementation Steps (Suggested Order)

1. **Step 1: Add API Function** (`frontend/lib/api.js`)
   - Guide user to add `getSessionHistory()` function
   - Explain API call pattern
   - Test API call manually first

2. **Step 2: Add State Management** (`frontend/hooks/useAdGenerator.js`)
   - Guide user to add history state variables
   - Guide user to add `fetchHistory()` function
   - Guide user to add `loadMoreHistory()` function
   - Guide user to add `resetHistory()` function

3. **Step 3: Create History Component** (`frontend/components/History.jsx`)
   - Guide user to create component structure
   - Guide user to implement event display (chat-style)
   - Guide user to add pagination controls
   - Guide user to add loading/error states

4. **Step 4: Integrate into Wizard** (`frontend/app/ad-generator/page.js`)
   - Guide user to decide placement (step/sidebar/modal)
   - Guide user to add History component
   - Guide user to connect props and state

5. **Step 5: Testing**
   - Guide user to test with existing session
   - Guide user to test pagination
   - Guide user to test error handling

### Key Considerations

- **Session ID**: History is session-specific. Only show history when `userSessionId` exists.
- **Pagination**: Backend returns 20 events per page by default. Frontend should handle "Load More" or page navigation.
- **Event Formatting**: Backend already formats events as strings with `\n` for line breaks. Frontend needs to handle display (CSS `white-space: pre-wrap` or convert `\n` to `<br>`).
- **Chronological Order**: Events are already ordered oldest-first by backend. Display in same order.
- **State Management**: History state should be separate from workflow state. Reset when new session starts.
- **Error Handling**: Handle cases where session doesn't exist, API fails, or no events exist.

### Questions to Clarify (If Needed)

- Where should History component be placed? (Step in wizard, sidebar, or modal?)
- Should history be automatically loaded when session starts, or only on demand?
- Should history persist across page refreshes, or reset on reload?
- What should happen to history when user starts a new session?

### Success Criteria

- ✅ History events display in chat-style format
- ✅ Pagination works (Load More / Previous Page)
- ✅ Loading states show during API calls
- ✅ Error states handle failures gracefully
- ✅ History integrates seamlessly with existing wizard UI
- ✅ Code follows existing patterns and style
- ✅ MVP-level implementation (simple, functional, not overcomplicated)

---

**Remember**: Guide step-by-step, explain concepts clearly, wait for approval, and let the user write the code. Your role is to teach and guide, not to implement.

