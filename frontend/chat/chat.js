const input = document.querySelector('#message');
const form = document.querySelector('#composer');
const send = document.querySelector('#send');
const reset = document.querySelector('#new-chat');
const messages = document.querySelector('#messages');
const welcome = document.querySelector('#welcome');
const error = document.querySelector('#error');
const connection = document.querySelector('#connection');
const conversation = document.querySelector('#conversation');
const journalCount = document.querySelector('#journal-count');
const journalStatus = document.querySelector('#journal-status');
const journalEntries = document.querySelector('#journal-entries');
const journalDescription = document.querySelector('#journal-description');
const loginForm = document.querySelector('#login-form');
const loginError = document.querySelector('#login-error');
const loginSubmit = document.querySelector('#login-submit');
const logout = document.querySelector('#logout');
let entryCount = 0;
let ready = false;
let busy = false;
let activeChatController = null;
const authChannel = typeof BroadcastChannel === 'function' ? new BroadcastChannel('reflect-auth') : null;
if (authChannel) authChannel.onmessage = () => {
  showLogin();
  // Other tabs may now belong to a different account: discard visible state.
  window.location.reload();
};

function controls() {
  input.disabled = !ready || busy;
  send.disabled = !ready || busy || !input.value.trim();
  reset.disabled = !ready || busy;
  logout.disabled = !ready;
  document.querySelectorAll('[data-prompt]').forEach(button => { button.disabled = !ready || busy; });
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const isChat = path === '/api/chat';
  if (isChat) activeChatController = controller;
  // Authentication/session requests must not leave the UI spinning forever.
  const timer = isChat ? null : setTimeout(() => controller.abort(), 15000);
  try {
    const response = await fetch(path, { credentials: 'same-origin', ...options, signal: controller.signal });
    if (!response.ok) {
      if (response.status === 401 && path !== '/api/login') showLogin();
      if (response.status === 401) {
        const failure = new Error(path === '/api/login' ? 'Invalid username or password.' : 'Your session ended. Please sign in.');
        failure.status = 401;
        throw failure;
      }
      if (response.status === 429) throw new Error(path === '/api/login'
        ? 'Too many sign-in attempts. Try again in a minute.'
        : 'The assistant is responding to another message. Please try again shortly.');
      if (response.status === 409) throw new Error('Your conversation changed while responding. Please try again.');
      throw new Error(response.status === 422 ? 'Please check your input and try again.' : 'Could not reach the assistant. Check that the Python server is running, then try again.');
    }
    return await response.json();
  } catch (failure) {
    if (failure.name === 'AbortError') throw new Error(isChat
      ? 'Response request cancelled.'
      : 'The server took too long to respond. Please check that startup has finished and try again.');
    throw failure;
  } finally {
    if (timer !== null) clearTimeout(timer);
    if (activeChatController === controller) activeChatController = null;
  }
}

function showError(message) { error.textContent = message; error.hidden = false; }

function showLogin() {
  ready = false;
  document.body.classList.add('logged-out');
  messages.replaceChildren();
  input.value = '';
  welcome.hidden = false;
  error.hidden = true;
  journalStatus.textContent = '';
  document.querySelector('#password').value = '';
  loginError.hidden = true;
  renderJournal({ count: 0, entries: [] });
  document.querySelector('#user-label').textContent = 'YOUR SPACE';
  controls();
}

loginForm.addEventListener('submit', async event => {
  event.preventDefault();
  loginSubmit.disabled = true;
  loginSubmit.textContent = 'Signing in…';
  loginError.hidden = true;
  try {
    await request('/api/login', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: document.querySelector('#username').value, password: document.querySelector('#password').value }),
    });
    document.querySelector('#password').value = '';
    authChannel?.postMessage('signed-in');
    await connect();
  } catch (failure) {
    loginError.textContent = failure.message;
    loginError.hidden = false;
    document.querySelector('#password').value = '';
  } finally { loginSubmit.disabled = false; loginSubmit.textContent = 'Sign in'; }
});

logout.addEventListener('click', async () => {
  try {
    await request('/api/logout', { method: 'POST' });
    authChannel?.postMessage('signed-out');
    activeChatController?.abort();
    busy = false;
    showLogin();
    document.querySelector('#username').focus();
  } catch (failure) { showError(failure.message); }
});

function renderJournal(data) {
  entryCount = data.count;
  journalCount.textContent = entryCount ? `${entryCount} ${entryCount === 1 ? 'entry' : 'entries'} available` : 'No entries yet';
  journalDescription.textContent = entryCount
    ? 'Your journal entries are here to help you reflect. We can explore a moment, notice a pattern, or simply talk about what stands out.'
    : 'No journal entries found. When you’re ready, write a few words about your day. A small moment is a lovely place to start.';
  if (data.entries.some(entry => entry.synthetic)) {
    journalDescription.textContent += ' Entries marked “demo” are fictional examples you can explore.';
  }
  journalEntries.replaceChildren();
  data.entries.forEach(entry => {
    const item = document.createElement('li');
    item.textContent = `${entry.date} · ${entry.title}${entry.synthetic ? ' (demo)' : ''}`;
    journalEntries.append(item);
  });
}



function addMessage(role, content, pending = false) {
  welcome.hidden = true;
  const turn = document.createElement('article');
  turn.className = `turn ${role}${pending ? ' pending' : ''}`;
  const label = document.createElement('p');
  label.className = 'turn-label';
  label.textContent = role === 'user' ? 'YOU' : 'REFLECT';
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.textContent = content;
  turn.append(label, bubble);
  messages.append(turn);
  conversation.scrollTop = conversation.scrollHeight;
  return turn;
}

function addSources(turn, sources) {
  const box = document.createElement('div');
  box.className = 'sources';
  sources.forEach((source, index) => {
    if (!source.url) return;
    let url;
    try { url = new URL(source.url); } catch { return; }
    if (!['http:', 'https:'].includes(url.protocol)) return;
    const link = document.createElement('a');
    link.href = url.href;
    link.target = '_blank';
    link.rel = 'noopener noreferrer';
    link.textContent = `[${index + 1}] ${source.title || 'Research source'}`;
    box.append(link);
  });
  if (box.childElementCount) turn.append(box);
}

async function connect() {
  try {
    const data = await request('/api/session');
    messages.replaceChildren();
    data.messages.forEach(message => addMessage(message.role, message.content));
    renderJournal(await request('/api/journal'));
    ready = true;
    document.body.classList.remove('logged-out');
    document.querySelector('#user-label').textContent = data.user.preferred_name || 'YOUR SPACE';
    connection.textContent = 'Local · Ready';
  } catch (failure) {
    if (document.body.classList.contains('logged-out')) {
      if (failure.status === 401) return;
      loginError.textContent = failure.message;
      loginError.hidden = false;
      return;
    }
    connection.textContent = 'Offline';
    showError('Could not connect. Start the Python server and reload this page.');
  }
  controls();
}

form.addEventListener('submit', async event => {
  event.preventDefault();
  const draft = input.value;
  const text = draft.trim();
  if (!ready || busy || !text) return;
  input.value = '';
  busy = true;
  error.hidden = true;
  controls();
  const userTurn = addMessage('user', text);
  const pending = addMessage('assistant', 'Thinking through your message…', true);
  const start = Date.now();
  const timer = setInterval(() => {
    pending.querySelector('.bubble').textContent = `Thinking through your message… ${Math.floor((Date.now() - start) / 1000)}s`;
  }, 1000);
  connection.textContent = 'Responding…';
  try {
    const data = await request('/api/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text }),
    });
    pending.remove();
    const turn = addMessage('assistant', data.answer);
    addSources(turn, data.sources || []);
    if (data.warnings?.length) {
      const details = document.createElement('details');
      const summary = document.createElement('summary');
      summary.textContent = 'Processing notes';
      const note = document.createElement('p');
      note.textContent = data.warnings.join('\n');
      details.append(summary, note);
      turn.append(details);
    }
  } catch (failure) {
    pending.remove();
    userTurn.remove();
    welcome.hidden = messages.childElementCount > 0;
    if (ready && !input.value) input.value = draft;
    showError(failure.message);
  } finally {
    clearInterval(timer);
    busy = false;
    connection.textContent = ready ? 'Local · Ready' : 'Sign in required';
    controls();
    input.focus();
    conversation.scrollTop = conversation.scrollHeight;
  }
});

input.addEventListener('input', controls);
input.addEventListener('keydown', event => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!send.disabled) form.requestSubmit();
  }
});
document.querySelectorAll('[data-prompt]').forEach(button => {
  button.addEventListener('click', () => { input.value = button.dataset.prompt; controls(); input.focus(); });
});
reset.addEventListener('click', async () => {
  if (!ready || busy) return;
  busy = true;
  controls();
  try {
    await request('/api/session', { method: 'POST' });
    messages.replaceChildren();
    welcome.hidden = false;
    input.value = '';
    error.hidden = true;
    renderJournal(await request('/api/journal'));
    journalStatus.textContent = '';
  } catch (failure) { showError(failure.message); }
  finally { busy = false; controls(); input.focus(); }
});
controls();
connect();
