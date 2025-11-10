"use client"

import { useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"

/**
 * History Component
 * 
 * Displays session history events in a chat-style format with pagination.
 * 
 * @param {string} userSessionId - Session ID (required to fetch history)
 * @param {Array<{ id: number | null, text: string, created_at: string | null }>} historyEvents - Array of history event objects
 * @param {number} historyPage - Current page number
 * @param {number} historyTotal - Total number of events
 * @param {number} historyLimit - Events per page
 * @param {boolean} historyHasMore - Whether more pages exist
 * @param {boolean} isLoadingHistory - Loading state
 * @param {Function} fetchHistory - Function to fetch history (page, limit)
 * @param {Function} loadMoreHistory - Function to load next page
 * @param {string|null} error - Error message (if any)
 */
export function History({
  userSessionId,
  historyEvents = [],
  historyPage = 1,
  historyTotal = 0,
  historyLimit = 20,
  historyHasMore = false,
  isLoadingHistory = false,
  fetchHistory,
  loadMoreHistory,
  error = null,
}) {
  // Automatically fetch history when component mounts and userSessionId exists
  useEffect(() => {
    if (userSessionId && historyEvents.length === 0 && !isLoadingHistory) {
      // Only fetch if we don't have any events yet and not already loading
      fetchHistory(1, historyLimit)
    }
  }, [userSessionId, fetchHistory, historyLimit]) // Only run when these change

  // Don't render if no session ID
  if (!userSessionId) {
    return null
  }

  // Calculate total pages for display
  const totalPages = Math.ceil(historyTotal / historyLimit)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-2xl">Session History</CardTitle>
        <CardDescription>
          View your workflow events chronologically
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-4">
          {/* Error Display */}
          {error && (
            <div className="p-4 bg-destructive/10 border border-destructive/20 rounded-lg">
              <p className="text-destructive text-sm">{error}</p>
            </div>
          )}

          {/* Loading State */}
          {isLoadingHistory && historyEvents.length === 0 && (
            <div className="p-8 text-center">
              <p className="text-muted-foreground">Loading history...</p>
            </div>
          )}

          {/* History Events Display */}
          {historyEvents.length > 0 ? (
            <div className="space-y-3">
              {/* Events List - Chat Style */}
              <div className="space-y-3 max-h-[600px] overflow-y-auto">
                {historyEvents.map((event, index) => {
                  const displayTime = event?.created_at
                    ? new Date(event.created_at).toLocaleString()
                    : null

                  return (
                    <div
                      key={event?.id ?? index}
                      className="bg-muted p-4 rounded-lg border"
                    >
                      <div className="flex items-start justify-between gap-3">
                        {/* Event Text - Preserve line breaks */}
                        <p className="text-foreground whitespace-pre-wrap text-sm flex-1">
                          {event?.text ?? ""}
                        </p>
                        {displayTime && (
                          <span className="text-xs text-muted-foreground whitespace-nowrap">
                            {displayTime}
                          </span>
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>

              {/* Pagination Info */}
              <div className="flex items-center justify-between text-sm text-muted-foreground pt-2 border-t">
                <div>
                  {historyTotal > 0 ? (
                    <span>
                      Showing {historyEvents.length} of {historyTotal} events
                      {totalPages > 1 && ` (Page ${historyPage} of ${totalPages})`}
                    </span>
                  ) : (
                    <span>No events to display</span>
                  )}
                </div>
              </div>

              {/* Load More Button */}
              {historyHasMore && (
                <div className="flex justify-center pt-2">
                  <Button
                    variant="outline"
                    onClick={loadMoreHistory}
                    disabled={isLoadingHistory}
                  >
                    {isLoadingHistory ? "Loading..." : "Load More"}
                  </Button>
                </div>
              )}

              {/* Loading More Indicator */}
              {isLoadingHistory && historyEvents.length > 0 && (
                <div className="text-center text-sm text-muted-foreground">
                  Loading more events...
                </div>
              )}
            </div>
          ) : (
            /* Empty State */
            !isLoadingHistory && (
              <div className="p-8 text-center">
                <p className="text-muted-foreground">
                  No history events found for this session.
                </p>
              </div>
            )
          )}
        </div>
      </CardContent>
    </Card>
  )
}

