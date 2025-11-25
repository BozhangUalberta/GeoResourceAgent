import TitleBar from "./TitleBar";
import MessageList from "./MessageList";
import MessageInput from "./MessageInput";

function MainWindow( { titleBarProps } ) {
  return (
    <div className="h-full overflow-y-auto [scrollbar-gutter:stable_both-edges]">

      <div className="h-10 w-full absolute left-0 top-0 z-15 px-3 pr-5">
        <TitleBar {...titleBarProps} />
      </div>
    
      <div className="w-full flex justify-center min-h-0 px-34">
        <MessageList />
      </div>

      <div className="h-auto w-full flex flex-col px-34
                      items-center justify-end absolute 
                      left-0 z-10 bottom-0 min-h-0 
                      bg-transparent">
        <MessageInput />
      </div>

    </div>
    
  );
}


export default MainWindow;