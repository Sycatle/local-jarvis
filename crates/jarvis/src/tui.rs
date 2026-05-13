//! Terminal UI client for the Jarvis D-Bus daemon. Spirit: Claude Code.
//!
//! Single binary, zero webview. Talks to `org.jarvis.Assistant` via the same
//! generic `zbus::Proxy` used by the rest of the CLI client module.

use std::collections::{HashSet, VecDeque};
use std::io::{stdout, Stdout};
use std::time::{Duration, Instant};

use anyhow::Result;
use crossterm::{
    event::{
        DisableMouseCapture, EnableMouseCapture, Event, EventStream, KeyCode, KeyEvent,
        KeyEventKind, KeyModifiers,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use futures::StreamExt;
use jarvis_service::{BUS_NAME, OBJECT_PATH};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Paragraph, Tabs, Wrap},
    Frame, Terminal,
};
use tokio::sync::mpsc;
use zbus::{Connection, Proxy};

const BUFFER_CAP: usize = 2000;
const RECONNECT_EVERY: Duration = Duration::from_secs(2);
const INTERFACE: &str = "org.jarvis.Assistant";

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Level {
    Info,
    Warn,
    Error,
    User,
    Jarv,
    Tool,
}

impl Level {
    fn tag(&self) -> &'static str {
        match self {
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERR ",
            Level::User => "USER",
            Level::Jarv => "JARV",
            Level::Tool => "TOOL",
        }
    }
    fn color(&self) -> Color {
        match self {
            Level::Info => Color::DarkGray,
            Level::Warn => Color::Yellow,
            Level::Error => Color::Red,
            Level::User => Color::Cyan,
            Level::Jarv => Color::Green,
            Level::Tool => Color::Magenta,
        }
    }
    fn parse(s: &str) -> Option<Level> {
        match s.to_ascii_uppercase().as_str() {
            "INFO" => Some(Level::Info),
            "WARN" => Some(Level::Warn),
            "ERROR" | "ERR" => Some(Level::Error),
            "USER" => Some(Level::User),
            "JARV" => Some(Level::Jarv),
            "TOOL" => Some(Level::Tool),
            _ => None,
        }
    }
    fn all() -> [Level; 6] {
        [
            Level::Info,
            Level::Warn,
            Level::Error,
            Level::User,
            Level::Jarv,
            Level::Tool,
        ]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tab {
    Session,
    Journal,
    Tools,
}

impl Tab {
    fn title(&self) -> &'static str {
        match self {
            Tab::Session => "Session",
            Tab::Journal => "Journal",
            Tab::Tools => "Tools",
        }
    }
    fn idx(&self) -> usize {
        match self {
            Tab::Session => 0,
            Tab::Journal => 1,
            Tab::Tools => 2,
        }
    }
    fn keeps(&self, lvl: Level) -> bool {
        match self {
            Tab::Session => true,
            Tab::Journal => matches!(lvl, Level::Info | Level::Warn | Level::Error),
            Tab::Tools => matches!(lvl, Level::Tool),
        }
    }
}

#[derive(Clone, Debug)]
struct LogLine {
    ts: Instant,
    level: Level,
    text: String,
}

#[derive(Debug)]
enum DaemonEvent {
    State(String),
    Transcribed(String),
    Spoken(String),
    Step {
        iteration: u32,
        thought: String,
        action: String,
        observation: String,
    },
    Connected(String),
    Disconnected,
}

struct App {
    lines: VecDeque<LogLine>,
    tab: Tab,
    silenced: HashSet<Level>,
    state: String,
    state_since: Instant,
    connected: bool,
    prompt: String,
    cursor: usize,
    busy: bool,
    scroll_off: usize, // lines from bottom; 0 = follow tail
    session_start: Instant,
    quit: bool,
}

impl App {
    fn new() -> Self {
        Self {
            lines: VecDeque::with_capacity(BUFFER_CAP),
            tab: Tab::Session,
            silenced: HashSet::new(),
            state: "disconnected".into(),
            state_since: Instant::now(),
            connected: false,
            prompt: String::new(),
            cursor: 0,
            busy: false,
            scroll_off: 0,
            session_start: Instant::now(),
            quit: false,
        }
    }

    fn append(&mut self, level: Level, text: impl Into<String>) {
        if self.lines.len() == BUFFER_CAP {
            self.lines.pop_front();
        }
        self.lines.push_back(LogLine {
            ts: Instant::now(),
            level,
            text: text.into(),
        });
    }

    fn visible(&self) -> Vec<&LogLine> {
        self.lines
            .iter()
            .filter(|l| self.tab.keeps(l.level) && !self.silenced.contains(&l.level))
            .collect()
    }

    fn apply_event(&mut self, ev: DaemonEvent) {
        match ev {
            DaemonEvent::Connected(state) => {
                if !self.connected {
                    self.connected = true;
                    self.state = state.clone();
                    self.state_since = Instant::now();
                    self.append(Level::Info, format!("dbus connected · state → {state}"));
                }
            }
            DaemonEvent::Disconnected => {
                if self.connected {
                    self.connected = false;
                    self.append(Level::Warn, "dbus disconnected");
                }
            }
            DaemonEvent::State(s) => {
                if !self.state.eq_ignore_ascii_case(&s) {
                    self.state = s.clone();
                    self.state_since = Instant::now();
                    self.append(Level::Info, format!("state → {s}"));
                }
                let lc = s.to_ascii_lowercase();
                self.busy = matches!(lc.as_str(), "listening" | "thinking" | "speaking");
            }
            DaemonEvent::Transcribed(t) => self.append(Level::User, t),
            DaemonEvent::Spoken(t) => self.append(Level::Jarv, t),
            DaemonEvent::Step {
                iteration,
                thought,
                action,
                observation,
            } => {
                if !thought.is_empty() {
                    self.append(Level::Tool, format!("#{iteration} thought: {thought}"));
                }
                if !action.is_empty() {
                    self.append(Level::Tool, format!("#{iteration} action: {action}"));
                }
                if !observation.is_empty() {
                    self.append(
                        Level::Tool,
                        format!("#{iteration} observation: {observation}"),
                    );
                }
            }
        }
    }

    fn run_command(&mut self, cmd: &str, args: &[&str]) -> Option<UiCommand> {
        match cmd {
            "help" => {
                self.append(
                    Level::Info,
                    "commands: :f <levels>  :listen  :cancel  :clear  :tab <1|2|3>  :quit  :help",
                );
                None
            }
            "f" | "filter" => {
                if args.is_empty() {
                    let sil: Vec<&str> = self.silenced.iter().map(|l| l.tag().trim()).collect();
                    self.append(
                        Level::Info,
                        format!(
                            "silenced: {}",
                            if sil.is_empty() {
                                "—".into()
                            } else {
                                sil.join(",")
                            }
                        ),
                    );
                    return None;
                }
                let mut keep = HashSet::new();
                for a in args {
                    for part in a.split(',') {
                        if let Some(l) = Level::parse(part.trim()) {
                            keep.insert(l);
                        }
                    }
                }
                self.silenced = Level::all()
                    .iter()
                    .copied()
                    .filter(|l| !keep.contains(l))
                    .collect();
                self.append(
                    Level::Info,
                    format!("filter set · showing: {}", args.join(",")),
                );
                None
            }
            "listen" => Some(UiCommand::Listen),
            "cancel" => Some(UiCommand::Cancel),
            "clear" => {
                self.lines.clear();
                None
            }
            "tab" => {
                match args.first().and_then(|s| s.parse::<u32>().ok()) {
                    Some(1) => self.tab = Tab::Session,
                    Some(2) => self.tab = Tab::Journal,
                    Some(3) => self.tab = Tab::Tools,
                    _ => self.append(Level::Warn, ":tab expects 1, 2 or 3"),
                }
                None
            }
            "quit" | "q" | "exit" => {
                self.quit = true;
                None
            }
            other => {
                self.append(Level::Warn, format!("unknown command :{other} · try :help"));
                None
            }
        }
    }
}

enum UiCommand {
    Ask(String),
    Listen,
    Cancel,
}

pub async fn run() -> Result<()> {
    let (ev_tx, mut ev_rx) = mpsc::channel::<DaemonEvent>(256);

    // D-Bus driver task: connect, subscribe to signals, reconnect on drop.
    let driver_tx = ev_tx.clone();
    let driver = tokio::spawn(async move { dbus_driver(driver_tx).await });

    // Command channel: prompts/commands from UI to D-Bus.
    let (cmd_tx, mut cmd_rx) = mpsc::channel::<UiCommand>(32);
    let dispatcher = tokio::spawn(async move {
        while let Some(cmd) = cmd_rx.recv().await {
            // each call is short-lived; ignore errors (status line surfaces disconnects)
            let _ = dispatch(cmd).await;
        }
    });

    let mut terminal = setup_terminal()?;
    let mut app = App::new();
    app.append(Level::Info, format!("daemon: connecting to {BUS_NAME}…"));

    let res = ui_loop(&mut terminal, &mut app, &mut ev_rx, &cmd_tx).await;

    restore_terminal(&mut terminal)?;
    drop(cmd_tx);
    driver.abort();
    let _ = dispatcher.await;
    let _ = driver.await;
    res
}

async fn ui_loop(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    app: &mut App,
    ev_rx: &mut mpsc::Receiver<DaemonEvent>,
    cmd_tx: &mpsc::Sender<UiCommand>,
) -> Result<()> {
    let mut keys = EventStream::new();
    let mut ticker = tokio::time::interval(Duration::from_millis(200));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        terminal.draw(|f| draw(f, app))?;
        if app.quit {
            return Ok(());
        }
        tokio::select! {
            biased;
            maybe_key = keys.next() => {
                if let Some(Ok(Event::Key(k))) = maybe_key { handle_key(app, k, cmd_tx).await }
            }
            maybe_ev = ev_rx.recv() => {
                if let Some(ev) = maybe_ev {
                    app.apply_event(ev);
                    // drain any other queued events without redrawing each one
                    while let Ok(ev) = ev_rx.try_recv() {
                        app.apply_event(ev);
                    }
                }
            }
            _ = ticker.tick() => {}
        }
    }
}

async fn handle_key(app: &mut App, k: KeyEvent, cmd_tx: &mpsc::Sender<UiCommand>) {
    if k.kind == KeyEventKind::Release {
        return;
    }
    let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
    match k.code {
        KeyCode::Char('c') if ctrl => app.quit = true,
        KeyCode::Char('d') if ctrl && app.prompt.is_empty() => app.quit = true,
        KeyCode::Char('l') if ctrl => app.lines.clear(),
        KeyCode::Char('1') if ctrl => app.tab = Tab::Session,
        KeyCode::Char('2') if ctrl => app.tab = Tab::Journal,
        KeyCode::Char('3') if ctrl => app.tab = Tab::Tools,
        KeyCode::Tab => {
            app.tab = match app.tab {
                Tab::Session => Tab::Journal,
                Tab::Journal => Tab::Tools,
                Tab::Tools => Tab::Session,
            }
        }
        KeyCode::PageUp => app.scroll_off = app.scroll_off.saturating_add(10),
        KeyCode::PageDown => app.scroll_off = app.scroll_off.saturating_sub(10),
        KeyCode::Home => app.scroll_off = usize::MAX / 2,
        KeyCode::End => app.scroll_off = 0,
        KeyCode::Esc if app.busy => {
            let _ = cmd_tx.try_send(UiCommand::Cancel);
        }
        KeyCode::Esc => app.quit = true,
        KeyCode::Left => {
            if app.cursor > 0 {
                app.cursor -= 1;
            }
        }
        KeyCode::Right => {
            if app.cursor < app.prompt.chars().count() {
                app.cursor += 1;
            }
        }
        KeyCode::Backspace => {
            if app.cursor > 0 {
                let byte = char_byte(&app.prompt, app.cursor - 1);
                let byte_end = char_byte(&app.prompt, app.cursor);
                app.prompt.replace_range(byte..byte_end, "");
                app.cursor -= 1;
            }
        }
        KeyCode::Delete => {
            let n = app.prompt.chars().count();
            if app.cursor < n {
                let byte = char_byte(&app.prompt, app.cursor);
                let byte_end = char_byte(&app.prompt, app.cursor + 1);
                app.prompt.replace_range(byte..byte_end, "");
            }
        }
        KeyCode::Enter => {
            let text = std::mem::take(&mut app.prompt);
            app.cursor = 0;
            let trimmed = text.trim();
            if trimmed.is_empty() {
                return;
            }
            if let Some(rest) = trimmed.strip_prefix(':') {
                let mut parts = rest.split_whitespace();
                let cmd = parts.next().unwrap_or("").to_string();
                let args: Vec<&str> = parts.collect();
                if let Some(action) = app.run_command(&cmd, &args) {
                    let _ = cmd_tx.try_send(action);
                }
            } else {
                app.append(Level::User, trimmed.to_string());
                app.busy = true;
                let _ = cmd_tx.try_send(UiCommand::Ask(trimmed.to_string()));
            }
        }
        KeyCode::Char(c) => {
            let byte = char_byte(&app.prompt, app.cursor);
            app.prompt.insert(byte, c);
            app.cursor += 1;
        }
        _ => {}
    }
}

fn char_byte(s: &str, char_idx: usize) -> usize {
    s.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(s.len())
}

fn draw(f: &mut Frame, app: &App) {
    let area = f.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // tabs
            Constraint::Length(1), // filter chips
            Constraint::Min(1),    // log
            Constraint::Length(3), // prompt
            Constraint::Length(1), // status
        ])
        .split(area);

    draw_tabs(f, chunks[0], app);
    draw_chips(f, chunks[1], app);
    draw_log(f, chunks[2], app);
    draw_prompt(f, chunks[3], app);
    draw_status(f, chunks[4], app);
}

fn draw_tabs(f: &mut Frame, area: Rect, app: &App) {
    let titles: Vec<Line> = [Tab::Session, Tab::Journal, Tab::Tools]
        .iter()
        .enumerate()
        .map(|(i, t)| Line::from(format!(" {} {} ", i + 1, t.title())))
        .collect();
    let tabs = Tabs::new(titles)
        .select(app.tab.idx())
        .style(Style::default().fg(Color::Gray))
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .divider("│");
    f.render_widget(tabs, area);
}

fn draw_chips(f: &mut Frame, area: Rect, app: &App) {
    let mut spans: Vec<Span> = Vec::new();
    spans.push(Span::styled(
        "filters ",
        Style::default().fg(Color::DarkGray),
    ));
    for lvl in Level::all() {
        let active = !app.silenced.contains(&lvl);
        let style = if active {
            Style::default()
                .fg(lvl.color())
                .add_modifier(Modifier::BOLD)
        } else {
            Style::default()
                .fg(Color::DarkGray)
                .add_modifier(Modifier::CROSSED_OUT)
        };
        spans.push(Span::styled(format!(" {} ", lvl.tag().trim()), style));
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

fn draw_log(f: &mut Frame, area: Rect, app: &App) {
    let visible = app.visible();
    let h = area.height as usize;
    let n = visible.len();
    let end = n.saturating_sub(app.scroll_off);
    let start = end.saturating_sub(h);
    let slice = &visible[start..end];
    let lines: Vec<Line> = slice
        .iter()
        .map(|l| {
            let elapsed = l.ts.saturating_duration_since(app.session_start);
            let secs = elapsed.as_secs();
            let ts = format!(
                "{:02}:{:02}:{:02}",
                secs / 3600,
                (secs / 60) % 60,
                secs % 60
            );
            Line::from(vec![
                Span::styled(format!("{ts} "), Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{} ", l.level.tag()),
                    Style::default()
                        .fg(l.level.color())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::raw(l.text.clone()),
            ])
        })
        .collect();
    let p = Paragraph::new(lines).wrap(Wrap { trim: false });
    f.render_widget(p, area);
}

fn draw_prompt(f: &mut Frame, area: Rect, app: &App) {
    let title = if app.busy {
        " prompt · busy "
    } else {
        " prompt "
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if app.busy {
            Color::Yellow
        } else {
            Color::DarkGray
        }))
        .title(title);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let display = if app.prompt.is_empty() {
        Span::styled(
            "type a message · :help for commands · Esc to quit",
            Style::default().fg(Color::DarkGray),
        )
    } else {
        Span::raw(app.prompt.clone())
    };
    f.render_widget(
        Paragraph::new(Line::from(vec![Span::raw("> "), display])),
        inner,
    );
    // place cursor (only when something is typed; else leave hidden)
    let cursor_col = inner.x + 2 + app.cursor as u16;
    if cursor_col < inner.x + inner.width {
        f.set_cursor_position((cursor_col, inner.y));
    }
}

fn draw_status(f: &mut Frame, area: Rect, app: &App) {
    let dot = if app.connected { "●" } else { "○" };
    let dot_style = Style::default().fg(if app.connected {
        Color::Green
    } else {
        Color::Red
    });
    let elapsed = app.state_since.elapsed().as_secs();
    let sil = if app.silenced.is_empty() {
        "—".to_string()
    } else {
        app.silenced
            .iter()
            .map(|l| l.tag().trim())
            .collect::<Vec<_>>()
            .join(",")
    };
    let line = Line::from(vec![
        Span::styled(format!("{dot} "), dot_style),
        Span::styled(
            format!("state={} ", app.state),
            Style::default()
                .fg(Color::White)
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(
            format!("· {elapsed}s "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            format!("· buf={}/{} ", app.lines.len(), BUFFER_CAP),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            format!("· silenced={sil} "),
            Style::default().fg(Color::DarkGray),
        ),
        Span::styled(
            format!("· tab={}", app.tab.title()),
            Style::default().fg(Color::DarkGray),
        ),
    ]);
    f.render_widget(Paragraph::new(line), area);
}

// ---- D-Bus side ------------------------------------------------------------

async fn dbus_driver(tx: mpsc::Sender<DaemonEvent>) {
    loop {
        match connect_and_subscribe(tx.clone()).await {
            Ok(()) => {
                // connect_and_subscribe blocks until a stream ends
                let _ = tx.send(DaemonEvent::Disconnected).await;
            }
            Err(_) => {
                let _ = tx.send(DaemonEvent::Disconnected).await;
            }
        }
        tokio::time::sleep(RECONNECT_EVERY).await;
    }
}

async fn connect_and_subscribe(tx: mpsc::Sender<DaemonEvent>) -> Result<()> {
    let conn = Connection::session().await?;
    let proxy = Proxy::new(&conn, BUS_NAME, OBJECT_PATH, INTERFACE).await?;
    let state: String = proxy
        .call_method("Status", &())
        .await?
        .body()
        .deserialize()?;
    tx.send(DaemonEvent::Connected(state)).await.ok();

    let mut state_s = proxy.receive_signal("StateChanged").await?;
    let mut trans_s = proxy.receive_signal("Transcribed").await?;
    let mut spoken_s = proxy.receive_signal("Spoken").await?;
    let mut step_s = proxy.receive_signal("StepTaken").await?;

    loop {
        tokio::select! {
            Some(m) = state_s.next() => {
                if let Ok(s) = m.body().deserialize::<String>() {
                    if tx.send(DaemonEvent::State(s)).await.is_err() { return Ok(()); }
                }
            }
            Some(m) = trans_s.next() => {
                if let Ok(s) = m.body().deserialize::<String>() {
                    if tx.send(DaemonEvent::Transcribed(s)).await.is_err() { return Ok(()); }
                }
            }
            Some(m) = spoken_s.next() => {
                if let Ok(s) = m.body().deserialize::<String>() {
                    if tx.send(DaemonEvent::Spoken(s)).await.is_err() { return Ok(()); }
                }
            }
            Some(m) = step_s.next() => {
                if let Ok((iteration, thought, action, observation)) =
                    m.body().deserialize::<(u32, String, String, String)>()
                {
                    if tx.send(DaemonEvent::Step { iteration, thought, action, observation }).await.is_err() {
                        return Ok(());
                    }
                }
            }
            else => return Ok(()),
        }
    }
}

async fn dispatch(cmd: UiCommand) -> Result<()> {
    let conn = Connection::session().await?;
    let proxy = Proxy::new(&conn, BUS_NAME, OBJECT_PATH, INTERFACE).await?;
    match cmd {
        UiCommand::Ask(text) => {
            proxy.call_method("Ask", &(text,)).await?;
        }
        UiCommand::Listen => {
            proxy.call_method("Listen", &()).await?;
        }
        UiCommand::Cancel => {
            proxy.call_method("Cancel", &()).await?;
        }
    }
    Ok(())
}

// ---- terminal lifecycle ----------------------------------------------------

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode()?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(out);
    Ok(Terminal::new(backend)?)
}

fn restore_terminal(t: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode()?;
    execute!(t.backend_mut(), LeaveAlternateScreen, DisableMouseCapture)?;
    t.show_cursor()?;
    Ok(())
}
