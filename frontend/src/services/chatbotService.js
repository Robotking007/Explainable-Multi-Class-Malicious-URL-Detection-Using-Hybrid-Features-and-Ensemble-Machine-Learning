import apiClient from './api';

export const chatbotService = {
  /**
   * Send question or prompt to the project's AI Security Assistant
   * @param {string} message - User query
   * @param {Array} history - Previous messages for context
   * @param {object} contextData - Current scan or URL context
   */
  async sendMessage(message, history = [], contextData = null) {
    if (!message || !message.trim()) {
      throw new Error('Message cannot be empty.');
    }

    const payload = {
      message: message.trim(),
      history: history.map(h => ({ role: h.role, content: h.content })),
      context: contextData || null
    };

    const candidateEndpoints = [
      `/api/chat`,
      `/api/assistant/query`,
      `/api/chatbot`,
      `/chat`
    ];

    for (const endpoint of candidateEndpoints) {
      try {
        const response = await apiClient.post(endpoint, payload);
        const reply =
          response.data?.response ||
          response.data?.message ||
          response.data?.reply ||
          response.data?.answer;

        if (reply) {
          return {
            role: 'assistant',
            content: reply,
            timestamp: new Date().toISOString(),
            sources: response.data?.sources || response.data?.references || []
          };
        }
      } catch (err) {
        if (err.status === 404) continue;
        throw err;
      }
    }

    throw new Error(
      'The AI Security Assistant endpoint is not currently reachable or implemented on the FastAPI backend. Check that `/api/chat` is exposed.'
    );
  },

  /**
   * Suggested standard questions for SOC analysts
   */
  getSuggestedQuestions() {
    return [
      "Why was this URL classified as malicious?",
      "What does this malware classification mean?",
      "Explain the top contributing features.",
      "What is ASN information?",
      "How does URL infrastructure graph analysis work?"
    ];
  }
};

export default chatbotService;
