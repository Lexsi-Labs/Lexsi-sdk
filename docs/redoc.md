# API Playground (Swagger UI)

Use the embedded Swagger UI to explore and call the Lexsi API. Replace the bearer token in the Authorize dialog with your `LEXSI_API_KEY`. Ensure CORS is allowed from this docs origin.

<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui.css">
<style>
  /* Light, high-contrast look within the dark site */
  .swagger-ui .topbar { display: none; }
  .swagger-ui .wrapper { background: #f5f7fb; padding: 16px; border-radius: 8px; }
  .swagger-ui .opblock { background: #f7fafd !important; border: 1px solid #c7d7f3 !important; }
  .swagger-ui .opblock-summary { background: #e9f1ff !important; color: #111 !important; border: 1px solid #c7d7f3 !important; }
  .swagger-ui .opblock-summary-method { background: #4a90e2 !important; color: #fff !important; }
  .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #51b37c !important; }
  .swagger-ui .btn, .swagger-ui .authorization__btn { background: #fff !important; color: #111 !important; border: 1px solid #c7d7f3 !important; }
  .swagger-ui .btn.authorize { background: #51b37c !important; color: #fff !important; border-color: #51b37c !important; }
  .swagger-ui .model-box { background: #fff !important; border: 1px solid #c7d7f3 !important; }
  .swagger-ui .response-col_description__inner div.markdown { color: #222 !important; }
  /* Force response sections to light */
  .swagger-ui .responses-wrapper,
  .swagger-ui .opblock-body,
  .swagger-ui .opblock-section { background: #f7fafd !important; }
  .swagger-ui .live-responses-table,
  .swagger-ui table.responses-table,
  .swagger-ui table.responses-table td,
  .swagger-ui table.responses-table th { background: #f7fafd !important; color: #111 !important; }
  .swagger-ui .curl-command,
  .swagger-ui .curl-command .curl { background: #ffffff !important; color: #111 !important; border: 1px solid #c7d7f3 !important; }
  .swagger-ui .highlight-code,
  .swagger-ui .response-col_description__inner pre,
  .swagger-ui .microlight { background: #ffffff !important; color: #111 !important; border: 1px solid #c7d7f3 !important; }
</style>
<div id="swagger-ui"></div>
<script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist/swagger-ui-bundle.js"></script>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    SwaggerUIBundle({
      url: "/openapi.yaml",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis],
      layout: "BaseLayout",
      docExpansion: "none",
      filter: true,
      tryItOutEnabled: true
    });
  });
</script>
