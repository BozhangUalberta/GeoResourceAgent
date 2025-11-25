import { http, HttpResponse } from 'msw'

const users = [];

let session = {};

function generateToken() {
  return "mock_jwt_token_" + Math.random().toString(36).slice(2);
}

function getUserByToken(token) {
  return session[token] || null;
}

function getAuthUser(request) {
  const auth = request.headers.get("authorization");
  if (!auth) return null;
  const token = auth.replace("Bearer ", "");
  const user = getUserByToken(token);
  return user || null;
}

// ================= Mock API ===================
const testConversationsList = Array.from({ length: 18 }, (_, i) => ({
  id: i,
  name: `whatever ${i}`,
}));

function testRunningTablesList(conversationId, runningTablesId) {
  return Array.from({ length: 18 }, (_, i) => ({
    id: i,
    name: `conversation ${conversationId} useful data ${i}`,}))
}


const testMessagesList = Array.from({ length: 28 }, (_, i) => ({
  id: i,
  name: i % 3 === 0 ? "system" : i % 3 === 1 ? "agent1" : "user",
  type: i % 3 === 0 ? "system" : i % 3 === 1 ? "agent" : "user",
  content: 
    i % 3 === 0
      ? `System notice ${i}`
      : i % 3 === 1
        ? `Agent1 reply ${i}`
        : `User input ${i}`,
}));


export const handlers = [
  // Login
  http.post("/api/v1/users/login", async ({ request }) => {    
    const body = await request.formData();
    const email = body.get("email");
    const password = body.get("password");
    
    const user = users.find(
      (u) => u.email === email && u.password === password
    );

    if (user) {
      const token = generateToken();
      session[token] = user;
      return HttpResponse.json({
        access_token: token,
        token_type: "bearer",
      });
    }
   
    return new HttpResponse("Incorrect email or password", { status: 401 });
  }),
  // Logout
  http.post("/api/v1/users/logout", async ({ request }) => {
    const auth = request.headers.get("authorization");
    const token = auth?.replace("Bearer ", "");
    if (token && session[token]) {
      delete session[token];
    }
    return new HttpResponse(null, { status: 204 });
  }),
  // Me
  http.get("/api/v1/users/me", async ({ request }) => {
    const auth = request.headers.get("authorization");
    const token = auth?.replace("Bearer ", "");
    const user = getUserByToken(token);
    if (user) {
      const { password, ...responseUser } = user;
      return HttpResponse.json(responseUser);
    }
    return new HttpResponse("Unauthorized", { status: 401 });
  }),
  // Register
  http.post("api/v1/users/register", async ({ request }) => {
    const { email, password } = await request.json();
    if (!email || !password)
      return new HttpResponse("Missing email or password", { status: 400 });
    if (users.find((u) => u.email === email)) {
      return new HttpResponse(
        JSON.stringify({ detail: "Email already registered" }),
        { status: 400, headers: { "Content-Type": "application/json" } }
      );
    }
     const newUser = {
      id: crypto.randomUUID(),
      email,
      password,
      full_name: email.split("@")[0],
      is_active: true,
      is_superuser: false,
      is_verified: false,
    };
    users.push(newUser);

    const { password: pw, ...responseUser } = newUser;
    return HttpResponse.json(responseUser);
  }),


  http.get("/api/v1/conversations", ({ request }) => {
    const user = getAuthUser(request);
    if (!user) return new HttpResponse("Unauthorized", { status: 401 });
    return HttpResponse.json(testConversationsList);
  }),
  http.get<{ conversationId: string, runningTablesId: string }>("/api/v1/conversations/:conversationId/running-tables/:runningTablesId", ({ request, params }) => {
    const user = getAuthUser(request);
    if (!user) return new HttpResponse("Unauthorized", { status: 401 });

    const { conversationId, runningTablesId } = params;
    return HttpResponse.json(testRunningTablesList(conversationId, runningTablesId));
  }),
  http.get<{ conversationId: string }>("/api/v1/conversations/:conversationId/messages", ({ request }) => {
    const user = getAuthUser(request);
    if (!user) return new HttpResponse("Unauthorized", { status: 401 });
    return HttpResponse.json(testMessagesList);
  }),
]